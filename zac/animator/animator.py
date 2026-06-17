from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches
import bisect
import os
import platform
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Dict, List, Optional, Tuple

MIN_FRAMES_FOR_PARALLEL = 60


@dataclass(frozen=True)
class FrameSnapshot:
    inst_str: str
    qubit_xs: Tuple[float, ...]
    qubit_ys: Tuple[float, ...]
    rydberg_sizes: Tuple[Tuple[float, float], ...]
    aod_col_x: Tuple[Tuple[float, ...], ...]
    aod_row_y: Tuple[Tuple[float, ...], ...]
    aod_col_colors: Tuple[Tuple[tuple, ...], ...]
    aod_row_colors: Tuple[Tuple[tuple, ...], ...]
    gate_positions: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class StaticRenderContext:
    slm_xs: Tuple[float, ...]
    slm_ys: Tuple[float, ...]
    figsize: Tuple[float, float]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    entanglement_rect_origins: Tuple[Tuple[float, float], ...]
    entanglement_rect_max_sizes: Tuple[Tuple[float, float], ...]
    aod_y_range: Tuple[float, float]
    aod_x_range: Tuple[float, float]
    aod_ids: Tuple[int, ...]
    aod_nc: Tuple[int, ...]
    aod_nr: Tuple[int, ...]
    font: int
    dpi: float
    frame_size: Tuple[int, int]


def _normalize_color(color) -> tuple:
    if isinstance(color, str):
        return color
    return tuple(float(c) for c in color)


def _resolve_ffmpeg_codec(
    ffmpeg: str = 'ffmpeg',
    codec: Optional[str] = None,
    use_gpu: bool = True,
) -> Tuple[str, List[str]]:
    if codec is None and use_gpu:
        codec = _detect_hw_codec(ffmpeg)
        if codec:
            print(f"[INFO] Animator: using GPU ffmpeg encoder {codec}")
        else:
            print("[INFO] Animator: no GPU ffmpeg encoder found, using software encoding")
    if codec is None:
        codec = 'h264'
    return codec, _ffmpeg_writer_extra_args(codec) if codec != 'h264' else ['-pix_fmt', 'yuv420p']


def _render_frame_task(
    frame_idx: int,
    snapshot: FrameSnapshot,
    ctx: StaticRenderContext,
) -> Tuple[int, bytes, Tuple[int, int]]:
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': ctx.font})
    fig, ax = plt.subplots(figsize=ctx.figsize, dpi=ctx.dpi)
    ax.set_xlim(ctx.xlim)
    ax.set_ylim(ctx.ylim)
    ax.scatter(
        ctx.slm_xs, ctx.slm_ys, marker='o', s=40,
        facecolor='none', edgecolor='g',
    )
    ax.scatter(
        snapshot.qubit_xs, snapshot.qubit_ys, marker='.', c='k',
    )
    for origin, (width, height) in zip(
        ctx.entanglement_rect_origins, snapshot.rydberg_sizes,
    ):
        rect = matplotlib.patches.Rectangle(
            origin, width, height,
            linewidth=1, edgecolor='none', facecolor=(0, 0, 1, 0.3),
        )
        ax.add_patch(rect)
    for aod_idx, _aod_id in enumerate(ctx.aod_ids):
        for col_id in range(ctx.aod_nc[aod_idx]):
            ax.axvline(
                snapshot.aod_col_x[aod_idx][col_id],
                ctx.aod_y_range[0], ctx.aod_y_range[1],
                c=snapshot.aod_col_colors[aod_idx][col_id],
                ls='--',
            )
        for row_id in range(ctx.aod_nr[aod_idx]):
            ax.axhline(
                snapshot.aod_row_y[aod_idx][row_id],
                ctx.aod_x_range[0], ctx.aod_x_range[1],
                c=snapshot.aod_row_colors[aod_idx][row_id],
                ls='--',
            )
    for x, y in snapshot.gate_positions:
        ax.scatter(x, y, s=300, color=(0, 1, 0, 0.5))
    ax.set_title(snapshot.inst_str)
    buffer = BytesIO()
    fig.savefig(buffer, format='raw', dpi=ctx.dpi)
    width, height = fig.canvas.get_width_height()
    plt.close(fig)
    return frame_idx, buffer.getvalue(), (width, height)


def _parallel_render_to_ffmpeg(
    states: List[FrameSnapshot],
    ctx: StaticRenderContext,
    workers: int,
    output: str,
    fps: int,
    ffmpeg: str,
    codec: str,
    extra_args: List[str],
    show_progress: bool,
) -> None:
    total = len(states)
    rendered: Dict[int, bytes] = {}
    next_to_write = 0
    progress = _make_animation_progress_callback() if show_progress else None
    max_in_flight = max(workers * 2, workers)
    width, height = ctx.frame_size
    command = [
        ffmpeg, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgba',
        '-framerate', str(fps), '-loglevel', 'error',
        '-i', 'pipe:',
        '-vcodec', codec,
    ] + extra_args + [output]
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    def drain_completed(done_futures) -> None:
        nonlocal next_to_write
        for future_done in done_futures:
            idx, frame_data, _size = future_done.result()
            rendered[idx] = frame_data
            while next_to_write in rendered:
                proc.stdin.write(rendered.pop(next_to_write))
                if progress is not None:
                    progress(next_to_write, total)
                next_to_write += 1

    with ProcessPoolExecutor(max_workers=workers) as executor:
        in_flight = {}
        for frame_idx in range(total):
            future = executor.submit(
                _render_frame_task, frame_idx, states[frame_idx], ctx,
            )
            in_flight[future] = frame_idx
            if len(in_flight) >= max_in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                drain_completed(done)
                for future_done in done:
                    del in_flight[future_done]
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            drain_completed(done)
            for future_done in done:
                del in_flight[future_done]

    proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr is not None else b''
    return_code = proc.wait()
    if return_code != 0:
        err = stderr.decode(errors='replace')
        raise subprocess.CalledProcessError(return_code, command, stderr=err)


def _list_ffmpeg_encoders(ffmpeg: str) -> str:
    try:
        proc = subprocess.run(
            [ffmpeg, '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return proc.stdout + proc.stderr


def _detect_hw_codec(ffmpeg: str = 'ffmpeg') -> Optional[str]:
    if not shutil.which(ffmpeg):
        return None

    encoders = _list_ffmpeg_encoders(ffmpeg)
    if not encoders:
        return None

    system = platform.system()
    if system == 'Darwin':
        candidates = ['h264_videotoolbox', 'hevc_videotoolbox']
    elif system == 'Linux':
        candidates = ['h264_nvenc', 'h264_vaapi', 'h264_qsv', 'hevc_nvenc']
    elif system == 'Windows':
        candidates = ['h264_nvenc', 'h264_qsv', 'h264_amf']
    else:
        candidates = ['h264_nvenc']

    for codec in candidates:
        for line in encoders.splitlines():
            if codec in line and line.strip().startswith('V'):
                return codec
    return None


def _ffmpeg_writer_extra_args(codec: str) -> List[str]:
    extra_args = ['-pix_fmt', 'yuv420p']
    if 'nvenc' in codec:
        extra_args.extend(['-preset', 'p4'])
    return extra_args


def _build_ffmpeg_writer(
    fps: int,
    ffmpeg: str = 'ffmpeg',
    codec: Optional[str] = None,
    use_gpu: bool = True,
) -> FFMpegWriter:
    resolved_codec, extra_args = _resolve_ffmpeg_codec(
        ffmpeg=ffmpeg, codec=codec, use_gpu=use_gpu,
    )
    if resolved_codec == 'h264':
        return FFMpegWriter(fps)
    return FFMpegWriter(
        fps,
        codec=resolved_codec,
        extra_args=extra_args,
    )


def _make_animation_progress_callback() -> Callable[[int, int], None]:
    state = {'last_pct': -1}

    def callback(current: int, total: int) -> None:
        if total is None or total <= 0:
            return
        done = current + 1
        pct = int(100 * done / total)
        if pct > state['last_pct'] or done == total:
            state['last_pct'] = pct
            print(
                f"\r[INFO] Animator: {done}/{total} ({pct}%)",
                end='',
                flush=True,
            )
            if done == total:
                print()

    return callback


class Animator():

    # constants for animation
    FPS = 60  # frames per second
    INIT_FRM = int(FPS / 5)  # initial empty frames, 1/5 second now
    PT_MICRON = 8  # scaling factor: points per micron
    MUS_PER_FRM = 150 / FPS  # microseconds per frame
    MUS_PER_FRM_SLOW = 7 / FPS # in slow motion, i.e., Rydberg
    CANVAS_PADDING = 10
    RYDBERG_PADDING = 3 # around each entanglement zone

    # colors
    RYDBERG_COLOR = 'b'
    SLM_COLOR = 'g'
    QUBIT_COLOR = 'k'
    AOD_COLORS = ['r', 'c', 'm', 'y'] # max 4 aods so far
    AOD_TRANS = 0.7


    def animate(self,
                 code: dict,
                 output: str,
                 scaling_factor: int = PT_MICRON,
                 font: int = 10,
                 ffmpeg: str = 'ffmpeg',
                 use_gpu: bool = True,
                 ffmpeg_codec: Optional[str] = None,
                 show_progress: bool = True,
                 parallel_workers: Optional[int] = None,
                 ):
        """
        Args:
            code (dict):
            output (str): filename to save output.
            scaling_factor (int, optional): the unit scaling factor between the
             animation and um. Defaults to PT_MICRON.
            font (int, optional): font size in the animation. Defaults to 10.
            ffmpeg (str, optional): path to the ffmpeg binary.
            use_gpu (bool, optional): auto-select a hardware ffmpeg encoder when
             available. Defaults to True.
            ffmpeg_codec (str, optional): explicit ffmpeg video codec, e.g.
             ``h264_videotoolbox`` or ``h264_nvenc``. Overrides ``use_gpu``.
            show_progress (bool, optional): print frame rendering progress.
             Defaults to True.
            parallel_workers (int, optional): number of worker processes for
             parallel frame rendering. ``None`` auto-selects based on CPU count.
             Use ``1`` to force sequential rendering.
        """

        matplotlib.use('Agg')
        matplotlib.rcParams.update({'font.size': font})
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg
        
        self.code = code
        self.fig, self.ax = self.setup_canvas(scaling_factor)
        self.title = self.ax.set_title('')
        self.inst_str = ''

        num_frame = self.create_schedule()
        total_frames = self.INIT_FRM + num_frame
        workers = self._resolve_parallel_workers(parallel_workers, total_frames)
        if show_progress:
            print(f"[INFO] Animator: rendering {total_frames} frames to {output}")
            if workers > 1:
                print(f"[INFO] Animator: parallel rendering with {workers} workers")

        if workers > 1:
            self._animate_parallel(
                output=output,
                total_frames=total_frames,
                scaling_factor=scaling_factor,
                font=font,
                ffmpeg=ffmpeg,
                use_gpu=use_gpu,
                ffmpeg_codec=ffmpeg_codec,
                show_progress=show_progress,
                workers=workers,
            )
        else:
            self._animate_sequential(
                output=output,
                total_frames=total_frames,
                ffmpeg=ffmpeg,
                use_gpu=use_gpu,
                ffmpeg_codec=ffmpeg_codec,
                show_progress=show_progress,
            )

    def _resolve_parallel_workers(
        self,
        parallel_workers: Optional[int],
        total_frames: int,
    ) -> int:
        if parallel_workers == 1:
            return 1
        workers = parallel_workers
        if workers is None:
            workers = min(os.cpu_count() or 4, 8)
        if total_frames < MIN_FRAMES_FOR_PARALLEL:
            return 1
        return max(1, workers)

    def _animate_sequential(
        self,
        output: str,
        total_frames: int,
        ffmpeg: str,
        use_gpu: bool,
        ffmpeg_codec: Optional[str],
        show_progress: bool,
    ) -> None:
        anim = FuncAnimation(
            self.fig,
            self.update,
            init_func=self.update_init,
            frames=total_frames,
        )
        writer = _build_ffmpeg_writer(
            self.FPS,
            ffmpeg=ffmpeg,
            codec=ffmpeg_codec,
            use_gpu=use_gpu,
        )
        save_kwargs = {}
        if show_progress:
            save_kwargs['progress_callback'] = _make_animation_progress_callback()
        anim.save(output, writer=writer, **save_kwargs)
        plt.close(self.fig)
        if show_progress:
            print(f"[INFO] Animator: saved {output}")

    def _animate_parallel(
        self,
        output: str,
        total_frames: int,
        scaling_factor: int,
        font: int,
        ffmpeg: str,
        use_gpu: bool,
        ffmpeg_codec: Optional[str],
        show_progress: bool,
        workers: int,
    ) -> None:
        self.update_init()
        static_ctx = self._build_static_render_context(scaling_factor, font)
        if show_progress:
            print(f"[INFO] Animator: computing {total_frames} frame states")
        states = self._precompute_frame_states(total_frames)
        plt.close(self.fig)

        codec, extra_args = _resolve_ffmpeg_codec(
            ffmpeg=ffmpeg, codec=ffmpeg_codec, use_gpu=use_gpu,
        )
        _parallel_render_to_ffmpeg(
            states,
            static_ctx,
            workers,
            output,
            self.FPS,
            ffmpeg,
            codec,
            extra_args,
            show_progress,
        )
        if show_progress:
            print(f"[INFO] Animator: saved {output}")

    def _build_static_render_context(
        self,
        scaling_factor: int,
        font: int,
    ) -> StaticRenderContext:
        slm_xs = []
        slm_ys = []
        for slm_id, slm_arr in self.architecture.dict_SLM.items():
            for r in range(slm_arr.n_r):
                for c in range(slm_arr.n_c):
                    x, y = self.architecture.exact_SLM_location(slm_id, r, c)
                    slm_xs.append(x)
                    slm_ys.append(y)
        dpi = float(plt.rcParams['figure.dpi'])
        px = 1 / dpi * scaling_factor
        figsize = self.fig.get_size_inches()
        width_px = int(round(figsize[0] * dpi))
        height_px = int(round(figsize[1] * dpi))
        aod_ids = tuple(sorted(self.architecture.dict_AOD))
        return StaticRenderContext(
            slm_xs=tuple(slm_xs),
            slm_ys=tuple(slm_ys),
            figsize=tuple(figsize),
            xlim=tuple(self.ax.get_xlim()),
            ylim=tuple(self.ax.get_ylim()),
            entanglement_rect_origins=tuple(
                zone[0] for zone in self.entanglement_rect_range
            ),
            entanglement_rect_max_sizes=tuple(
                (zone[1], zone[2]) for zone in self.entanglement_rect_range
            ),
            aod_y_range=(
                self.architecture.arch_range[0][1],
                self.architecture.arch_range[1][1],
            ),
            aod_x_range=(
                self.architecture.arch_range[0][0],
                self.architecture.arch_range[1][0],
            ),
            aod_ids=aod_ids,
            aod_nc=tuple(
                self.architecture.dict_AOD[aod_id].n_c for aod_id in aod_ids
            ),
            aod_nr=tuple(
                self.architecture.dict_AOD[aod_id].n_r for aod_id in aod_ids
            ),
            font=font,
            dpi=dpi,
            frame_size=(width_px, height_px),
        )

    def _precompute_frame_states(self, total_frames: int) -> List[FrameSnapshot]:
        states = []
        for frame_idx in range(total_frames):
            self.update(frame_idx)
            states.append(self._capture_frame_state())
        return states

    def _capture_frame_state(self) -> FrameSnapshot:
        aod_ids = tuple(sorted(self.architecture.dict_AOD))
        aod_col_x = []
        aod_row_y = []
        aod_col_colors = []
        aod_row_colors = []
        for aod_id in aod_ids:
            aod_col_x.append(tuple(
                float(line.get_xdata()[0]) for line in self.aod_col_plots[aod_id]
            ))
            aod_row_y.append(tuple(
                float(line.get_ydata()[0]) for line in self.aod_row_plots[aod_id]
            ))
            aod_col_colors.append(tuple(
                _normalize_color(line.get_color())
                for line in self.aod_col_plots[aod_id]
            ))
            aod_row_colors.append(tuple(
                _normalize_color(line.get_color())
                for line in self.aod_row_plots[aod_id]
            ))
        gate_positions = []
        for gate_idx in range(self._1q_gate_used):
            gate = self.qubit_1qGate[gate_idx]
            if gate.get_visible():
                x, y = gate.get_offsets()[0]
                gate_positions.append((float(x), float(y)))
        return FrameSnapshot(
            inst_str=self.inst_str,
            qubit_xs=tuple(self.qubit_xs),
            qubit_ys=tuple(self.qubit_ys),
            rydberg_sizes=tuple(
                (rect.get_width(), rect.get_height())
                for rect in self.entanglemet_rect
            ),
            aod_col_x=tuple(aod_col_x),
            aod_row_y=tuple(aod_row_y),
            aod_col_colors=tuple(aod_col_colors),
            aod_row_colors=tuple(aod_row_colors),
            gate_positions=tuple(gate_positions),
        )

    def create_schedule(self):
        """
        each frame is a sample on the time axis. There are two sampling rates
        one is regular, one is slow motion. The latter is used when Rydberg
        is happening because Rydberg is so fast that it won't appear in the
        video is using the regular sampling rate.

        The time axis looks like this:

        |_____________|...|_______________|...|_________|...|__________|

        where each | will be an entry in self.piecewise_schedule. The _ stands
        for regular period, the . stands for slow motion periods. 

        In piecewise_schedule consists of 3-tuples. the first number is the
        frame at the | The second number is whether the period before the | is
        slow motion (1) or not (0). The third number is the real time at the | 
        """

        self.piecewise_schedule = [(0, 0, 0), ] # add the first trivial entry
        last_end_time = 0
        for inst in self.code["instructions"]:
            if inst["type"] == "rydberg":

                # add the entry corresponding to the regular period before
                last_end_frame = self.piecewise_schedule[-1][0]
                self.piecewise_schedule.append(
                    (
                        last_end_frame + round(
                            (
                                inst["begin_time"] - last_end_time
                                ) / self.MUS_PER_FRM),
                        0,
                        inst["begin_time"]
                    )
                )

                # add the entry corresponding to the slow period for this inst
                last_end_frame = self.piecewise_schedule[-1][0]
                self.piecewise_schedule.append(
                    (
                        last_end_frame + round(
                            (
                                inst["end_time"] - inst["begin_time"]
                                ) / self.MUS_PER_FRM_SLOW),
                        1,
                        inst["end_time"]
                    )
                )
                last_end_time = inst["end_time"]

        # add an entry of the left over runtime after the last rydberg
        if self.code["runtime"] > last_end_time:
            last_end_frame = self.piecewise_schedule[-1][0]
            self.piecewise_schedule.append(
                    (
                        last_end_frame + round(
                            (
                                self.code["runtime"] - last_end_time
                                ) / self.MUS_PER_FRM),
                        0,
                        self.code["runtime"]
                    )
                )
        self._schedule_interval_ends = [
            interval[0] for interval in self.piecewise_schedule
        ]
        return self.piecewise_schedule[-1][0]

    def setup_canvas(self, scaling_factor: int):
        """set up various objects before actually drawing."""

        # arch_range is [[bottom_left x,y], [top_right x,y]]
        # unit conversion factor from um to pt
        px = 1/plt.rcParams['figure.dpi'] * scaling_factor
        fig, ax, = plt.subplots(
            figsize=(
                (2*self.CANVAS_PADDING + self.architecture.arch_range[1][0] - \
                 self.architecture.arch_range[0][0]) * px,
                (2*self.CANVAS_PADDING + self.architecture.arch_range[1][1] - \
                 self.architecture.arch_range[0][1]) * px
            )
        )
        ax.set_xlim([
            -self.CANVAS_PADDING + self.architecture.arch_range[0][0],
            self.CANVAS_PADDING + self.architecture.arch_range[1][0]
            ])
        ax.set_ylim([
            -self.CANVAS_PADDING + self.architecture.arch_range[0][1],
            self.CANVAS_PADDING + self.architecture.arch_range[1][1]
            ])
        
        # rydberg_range is a list. Each entry is for an entanglement zone,
        # each entry is a pair [[bottom_left x, y], [top_right x,y]]
        # self.entanglement_rect_range is for matplotlib plotting. the first
        # entry is the bottom_left x,y (with padding). The second entry is
        # the width, and the third entry is the height.
        self.entanglement_rect_range = [
            (
                (
                    range_pair[0][0] - self.RYDBERG_PADDING,
                    range_pair[0][1] - self.RYDBERG_PADDING
                ),
                range_pair[1][0] - range_pair[0][0] + 2 * self.RYDBERG_PADDING,
                range_pair[1][1] - range_pair[0][1] + 2 * self.RYDBERG_PADDING,
            )
            for range_pair in self.architecture.rydberg_range
        ]

        return fig, ax

    def update_init(self):
        # find all slms
        slm_xs = []
        slm_ys = []
        for slm_id, slm_arr in self.architecture.dict_SLM.items():
            for r in range(slm_arr.n_r):
                for c in range(slm_arr.n_c):
                    x, y = self.architecture.exact_SLM_location(slm_id, r, c)
                    slm_xs.append(x)
                    slm_ys.append(y)
        # draw slms
        self.ax.scatter(
            slm_xs, slm_ys, marker='o', s=40, facecolor='none',
            edgecolor=self.SLM_COLOR
        )

        # initialize qubits
        self.qubit_xs = []
        self.qubit_ys = []
        for q in self.code["instructions"][0]["init_locs"]:
            x, y = self.architecture.exact_SLM_location(q[1], q[2], q[3])
            self.qubit_xs.append(x)
            self.qubit_ys.append(y)
        # draw qubits
        self.qubit_scat = self.ax.scatter(
            self.qubit_xs, self.qubit_ys, marker='.', c=self.QUBIT_COLOR)

        # initialize aod cols
        self.aod_col_plots = {
            aod_id: [
                self.ax.axvline(
                    0,
                    self.architecture.arch_range[0][1],
                    self.architecture.arch_range[1][1],
                    c=(0,0,0,0),
                    ls='--'
                ) for _ in range(aod.n_c)
            ] for aod_id, aod in self.architecture.dict_AOD.items()
        }

        # initilize aod rows
        self.aod_row_plots = {
            aod_id: [
                self.ax.axhline(
                    0,
                    self.architecture.arch_range[0][0],
                    self.architecture.arch_range[1][0],
                    c=(0,0,0,0),
                    ls='--'
                ) for _ in range(aod.n_r)
            ] for aod_id, aod in self.architecture.dict_AOD.items()
        }

        # initialize Rydberg zones
        self.entanglemet_rect = []
        for entangle_zone in self.entanglement_rect_range:
            rect =  matplotlib.patches.Rectangle(
                entangle_zone[0],
                entangle_zone[1],
                entangle_zone[2],
                linewidth=1,
                edgecolor='none',
                # facecolor=(self.RYDBERG_COLOR, 0.3) # !
                facecolor=(0, 0, 1, 0.3) #
            )
            self.ax.add_patch(rect)
            self.entanglemet_rect.append(rect)

        # initialize single qubit gates
        self.qubit_1qGate = []
        self._1q_gate_used = 0
        self._anim_instructions = self.code["instructions"][1:]
        self._dirty_aod_cols = set()
        self._dirty_aod_rows = set()
        self._dirty_rydberg_zones = set()
        self._qubit_offsets_dirty = False
        return

    def _reset_dirty_rydberg_zones(self):
        for zone_id in self._dirty_rydberg_zones:
            self.entanglemet_rect[zone_id].set_width(0)
            self.entanglemet_rect[zone_id].set_height(0)
        self._dirty_rydberg_zones.clear()

    def _reset_dirty_aod_lines(self):
        for aod_id, col_id in self._dirty_aod_cols:
            self.aod_col_plots[aod_id][col_id].set_color((0, 0, 0, 0))
        for aod_id, row_id in self._dirty_aod_rows:
            self.aod_row_plots[aod_id][row_id].set_color((0, 0, 0, 0))
        self._dirty_aod_cols.clear()
        self._dirty_aod_rows.clear()

    def _hide_1q_gates(self):
        for gate in self.qubit_1qGate:
            gate.set_visible(False)
        self._1q_gate_used = 0

    def _set_aod_col_color(self, aod_id: int, col_id: int, color):
        self.aod_col_plots[aod_id][col_id].set_color(color)
        self._dirty_aod_cols.add((aod_id, col_id))

    def _set_aod_row_color(self, aod_id: int, row_id: int, color):
        self.aod_row_plots[aod_id][row_id].set_color(color)
        self._dirty_aod_rows.add((aod_id, row_id))

    def _iter_active_instructions(self, true_time: float):
        for inst in self._anim_instructions:
            if true_time >= inst["begin_time"] and true_time < inst["end_time"]:
                yield inst

    def update(self, f: int):  # f is the frame
        true_frame = f - self.INIT_FRM # consider the initial frozen frames

        # get which piecewise schedule f is in
        index = bisect.bisect_right(self._schedule_interval_ends, true_frame)
        tmp = self.piecewise_schedule[index]
        # calculate true time of this frame: tmp[2] is the end time of this
        # period. tmp[0] is the end frame of this period. So we deduct the 
        # remianing time from tmp[2]. The remaining time is calculated as the
        # product of remaining frames=tmp[0]-true_frame, and the sampling rate
        # which depends on whether this period is regular or slow motion.
        true_time = tmp[2] - (
            tmp[0] - true_frame) * (
                self.MUS_PER_FRM_SLOW if tmp[1] else self.MUS_PER_FRM)

        self.inst_str = ''
        self._qubit_offsets_dirty = False
        self._reset_dirty_rydberg_zones()
        self._hide_1q_gates()
        self._reset_dirty_aod_lines()
        
        if f >= self.INIT_FRM:
            for inst in self._iter_active_instructions(true_time):
                if inst["type"] == "rydberg":
                    self.update_rydberg(inst)
                elif inst["type"] == "rearrangeJob":
                    self.update_arrangement(true_time, inst)
                elif inst['type'] == '1qGate':
                    self.update_1qGate(inst)
                else:
                    raise ValueError(f"unknown inst type {inst['type']}")

        if self._qubit_offsets_dirty:
            self.qubit_scat.set_offsets(
                list(zip(self.qubit_xs, self.qubit_ys)))

        self.title.set_text(self.inst_str)
        return

    def update_rydberg(self, inst: dict):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["begin_time"]:.2f}'
        zone_id = inst["zone_id"]
        self.entanglemet_rect[zone_id].set_width(
            self.entanglement_rect_range[zone_id][1]
        )
        self.entanglemet_rect[zone_id].set_height(
            self.entanglement_rect_range[zone_id][2]
        )
        self._dirty_rydberg_zones.add(zone_id)

    def update_arrangement(self, time: float, inst: dict):
        self.inst_str += f' | {inst["id"]} {inst["type"]}'
        for detail_inst in inst["insts"]:
            if detail_inst["begin_time"] > time:
                break
            if time < detail_inst["end_time"]:
                ratio = (time - detail_inst["begin_time"]) / (
                    detail_inst["end_time"] - detail_inst["begin_time"])
                if detail_inst["type"] == "activate":
                    return self.update_activate(
                        ratio, time, detail_inst, inst["aod_id"])
                elif detail_inst["type"] == "deactivate":
                    return self.update_deactivate(
                        ratio, time, detail_inst, inst["aod_id"])
                elif detail_inst["type"].startswith("move"):
                    return self.update_move(
                        ratio,
                        time,
                        detail_inst,
                        zip(
                            detail_inst["begin_coord"],
                            detail_inst["end_coord"],
                        ),
                        inst["aod_id"],
                    )
        
    def update_activate(self, ratio: float, time: float, inst: dict, aod_id: int):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'
        for col_id, col_x in zip(inst["col_id"], inst["col_x"]):
            self.aod_col_plots[aod_id][col_id].set_xdata((col_x, ))
            self._set_aod_col_color(
                aod_id, col_id, (1, 0, 0, ratio*self.AOD_TRANS))
        for row_id, row_y in zip(inst["row_id"], inst["row_y"]):
            self.aod_row_plots[aod_id][row_id].set_ydata((row_y, ))
            self._set_aod_row_color(
                aod_id, row_id, (1, 0, 0, ratio*self.AOD_TRANS))

    def update_deactivate(self, ratio: float, time: float, inst: dict, aod_id: int):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'
        for col_id in inst["col_id"]:
            self._set_aod_col_color(
                aod_id, col_id, (1, 0, 0, (1-ratio)*self.AOD_TRANS))
        for row_id in inst["row_id"]:
            self._set_aod_row_color(
                aod_id, row_id, (1, 0, 0, (1-ratio)*self.AOD_TRANS))

    def update_move(self, ratio: float, time: float, inst: dict, qubit_coord, aod_id: int):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {time:.2f}'
        
        def interpolate(r: float, begin: int, end: int):
            D = end - begin
            return begin + 3*D*(r**2) - 2*D*(r**3)
        
        # update qubit
        for begin_coords_row, end_coords_row in qubit_coord:
            for begin_coords, end_coords in zip(
                begin_coords_row, end_coords_row):
                q_id = begin_coords["id"]
                self.qubit_xs[q_id] = interpolate(
                    ratio, begin_coords["x"], end_coords["x"])
                self.qubit_ys[q_id] = interpolate(
                    ratio, begin_coords["y"], end_coords["y"])
                self._qubit_offsets_dirty = True

        # update AOD
        for row_id, row_begin_y, row_end_y in zip(
            inst["row_id"], inst["row_y_begin"], inst["row_y_end"]):
            self.aod_row_plots[aod_id][row_id].set_ydata(
                (interpolate(ratio, row_begin_y, row_end_y), ))
            self._set_aod_row_color(aod_id, row_id, self.AOD_COLORS[aod_id])
        for col_id, col_begin_x, col_end_x in zip(
            inst["col_id"], inst["col_x_begin"], inst["col_x_end"]):
            self.aod_col_plots[aod_id][col_id].set_xdata(
                (interpolate(ratio, col_begin_x, col_end_x), ))
            self._set_aod_col_color(aod_id, col_id, self.AOD_COLORS[aod_id])
    
    def update_1qGate(self, inst: dict):
        self.inst_str += f' | {inst["id"]} {inst["type"]} \n elapsed time: {inst["end_time"]:.2f}'

        for g in inst['gates']:
            q = g['q']
            x = self.qubit_xs[q]
            y = self.qubit_ys[q]
            if self._1q_gate_used < len(self.qubit_1qGate):
                gate = self.qubit_1qGate[self._1q_gate_used]
                gate.set_offsets([(x, y)])
                gate.set_visible(True)
            else:
                gate = self.ax.scatter(x, y, s=300, color=(0, 1, 0, 0.5))
                self.qubit_1qGate.append(gate)
            self._1q_gate_used += 1