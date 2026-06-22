from zac.ds.architecture import Architecture
import numpy as np
import math
from copy import deepcopy
import gc

class PointEmbeddingPlacer:
    """class to find a qubit layout via point-grid embedding."""

    def __init__(self):
        self.home_storage_mapping = None

    # return G_i
    # architecture.entanglement_zone is a list of SLM id list. shape: [[id1, id2]]
    # use these id to find SLM objects in architecture.dict_SLM 
    # list_gate: gates in one layer, each gate is a 2-tuple
    # storage_mapping[i] is the location of the ith qubit, in the format of (slm_id, y, x)
    def place_gate(self, architecture: Architecture, list_gate: [[int, int]], storage_mapping: [(int, int, int)]) -> list:
        # get entanglement zone width and height
        slmid_l, slmid_r = architecture.entanglement_zone[0]
        slm0 = architecture.dict_SLM[slmid_l]
        ezh, ezw = slm0.n_r, slm0.n_c
        entanglement_slms = {slm_id for zone in architecture.entanglement_zone for slm_id in zone}
        active_slms = {slmid_l, slmid_r}

        if self.home_storage_mapping is None:
            self.home_storage_mapping = deepcopy(storage_mapping)

        K = len(list_gate)
        assert ezh * ezw >= K

        occupied_cells = set()
        for loc in storage_mapping:
            if loc[0] in entanglement_slms:
                if loc[0] not in active_slms:
                    raise ValueError("point embedding currently supports one entanglement-zone SLM pair")
                cell = (loc[2], loc[1])
                if not (0 <= cell[0] < ezw and 0 <= cell[1] < ezh):
                    raise ValueError("existing entanglement-zone qubit is outside the active grid")
                occupied_cells.add(cell)

        midpoints = []
        midpoint_gate_indices = []
        fixed_gate_cells = {}
        for gate_index, (q1, q2) in enumerate(list_gate):
            loc1, loc2 = storage_mapping[q1], storage_mapping[q2]
            q1_in_ez = loc1[0] in entanglement_slms
            q2_in_ez = loc2[0] in entanglement_slms
            if q1_in_ez or q2_in_ez:
                fixed_cells = []
                if q1_in_ez:
                    fixed_cells.append((loc1[2], loc1[1]))
                if q2_in_ez:
                    fixed_cells.append((loc2[2], loc2[1]))
                if any(cell != fixed_cells[0] for cell in fixed_cells):
                    raise ValueError("reused qubits of one gate occupy different interaction cells")
                fixed_gate_cells[gate_index] = fixed_cells[0]
                continue

            midy = loc1[1] + loc2[1] # no need to divide by 2, since we only need the relative location
            midx = loc1[2] + loc2[2]
            midpoints.append((midx, midy))
            midpoint_gate_indices.append(gate_index)

        if len(midpoints) + len(occupied_cells) > ezh * ezw:
            raise ValueError("not enough free entanglement-zone cells for non-reused gates")

        midpoint_legalized = self._embed_midpoints(midpoints, ezh, ezw, occupied_cells)
        gate_cells = dict(fixed_gate_cells)
        for gate_index, cell in zip(midpoint_gate_indices, midpoint_legalized):
            gate_cells[gate_index] = cell

        # full mapping generation
        gate_mapping = deepcopy(storage_mapping)
        for gate_index, (q1, q2) in enumerate(list_gate):
            x, y = gate_cells[gate_index]
            loc1, loc2 = storage_mapping[q1], storage_mapping[q2]
            q1_in_ez = loc1[0] in entanglement_slms
            q2_in_ez = loc2[0] in entanglement_slms
            if q1_in_ez and q2_in_ez:
                continue
            if q1_in_ez:
                gate_mapping[q2] = (slmid_r if loc1[0] == slmid_l else slmid_l, y, x)
                continue
            if q2_in_ez:
                gate_mapping[q1] = (slmid_r if loc2[0] == slmid_l else slmid_l, y, x)
                continue

            q1_x = loc1[2]
            q2_x = loc2[2]
            if q1_x > q2_x:
                q1, q2 = q2, q1
            gate_mapping[q1] = (slmid_l, y, x)
            gate_mapping[q2] = (slmid_r, y, x)

        return gate_mapping

    def choose_next_storage_and_gate(
        self,
        architecture: Architecture,
        current_gate_mapping: list,
        next_gate: list,
        reuse_qubits: set,
    ) -> tuple[list, list, bool]:
        reuse_qubits = set() if reuse_qubits is None else set(reuse_qubits)

        no_reuse_storage = deepcopy(self.home_storage_mapping)
        no_reuse_gate = self.place_gate(architecture, next_gate, no_reuse_storage)
        no_reuse_score = (
            self._strict_inversion_score(architecture, current_gate_mapping, no_reuse_storage)
            + self._strict_inversion_score(architecture, no_reuse_storage, no_reuse_gate)
        )

        if not reuse_qubits:
            return no_reuse_storage, no_reuse_gate, False

        reuse_storage = deepcopy(self.home_storage_mapping)
        for q in reuse_qubits:
            reuse_storage[q] = current_gate_mapping[q]
        try:
            reuse_gate = self.place_gate(architecture, next_gate, reuse_storage)
        except ValueError:
            return no_reuse_storage, no_reuse_gate, False
        reuse_score = (
            self._strict_inversion_score(architecture, current_gate_mapping, reuse_storage)
            + self._strict_inversion_score(architecture, reuse_storage, reuse_gate)
        )

        if reuse_score <= no_reuse_score:
            return reuse_storage, reuse_gate, True
        return no_reuse_storage, no_reuse_gate, False

    def final_storage_mapping(self) -> list:
        return deepcopy(self.home_storage_mapping)

    def run(self, architecture: Architecture, list_gate: [[int, int]], storage_mapping: [(int, int, int)], reuse_qubits: {int}) -> (list, list):
        gate_mapping = self.place_gate(architecture, list_gate, storage_mapping)
        next_storage_mapping = deepcopy(self.home_storage_mapping)
        reuse_qubits = set() if reuse_qubits is None else set(reuse_qubits)
        for q in reuse_qubits:
            next_storage_mapping[q] = gate_mapping[q]
        return gate_mapping, next_storage_mapping

    def _strict_inversion_score(self, architecture: Architecture, start_mapping: list, end_mapping: list) -> int:
        moved = [q for q in range(len(start_mapping)) if start_mapping[q] != end_mapping[q]]
        if len(moved) < 2:
            return 0

        start_x = []
        start_y = []
        end_x = []
        end_y = []
        for q in moved:
            sx, sy = architecture.exact_SLM_location_tuple(start_mapping[q])
            ex, ey = architecture.exact_SLM_location_tuple(end_mapping[q])
            start_x.append(sx)
            start_y.append(sy)
            end_x.append(ex)
            end_y.append(ey)

        return (
            self._count_strict_1d_inversions(start_x, end_x)
            + self._count_strict_1d_inversions(start_y, end_y)
        )

    def _count_strict_1d_inversions(self, start_coords: list, end_coords: list) -> int:
        if len(start_coords) < 2:
            return 0

        end_rank = {value: rank for rank, value in enumerate(sorted(set(end_coords)))}
        pairs = sorted((start, end_rank[end]) for start, end in zip(start_coords, end_coords))
        fenwick = _FenwickTree(len(end_rank))
        inversions = 0
        seen = 0
        i = 0
        while i < len(pairs):
            j = i + 1
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1

            for _, rank in pairs[i:j]:
                inversions += seen - fenwick.prefix_sum(rank)
            for _, rank in pairs[i:j]:
                fenwick.add(rank, 1)
                seen += 1
            i = j

        return inversions

    def _embed_midpoints(
        self,
        midpoints: list[tuple[int, int]],
        rows: int,
        cols: int,
        occupied_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not midpoints:
            return []

        source_rows, source_cols, midpoints_ranked = self._rank_compress(midpoints)

        rank_grid_fits = source_rows <= rows and source_cols <= cols
        ranked_points_unique = len(set(midpoints_ranked)) == len(midpoints_ranked)
        if rank_grid_fits and ranked_points_unique:
            midpoint_legalized = self._place_ranked_grid_middle_bottom(
                midpoints_ranked,
                source_rows,
                source_cols,
                rows,
                cols,
            )
            if self._cells_are_available(midpoint_legalized, occupied_cells):
                return midpoint_legalized
            return self._point_embedding_nearest_free(midpoint_legalized, rows, cols, occupied_cells)

        midpoint_transformed = self._affine_transform(midpoints_ranked, source_rows, source_cols, rows, cols)
        midpoint_legalized = self._point_embedding_nearest_free(midpoint_transformed, rows, cols, occupied_cells)
        if occupied_cells:
            return midpoint_legalized

        compact_rows, compact_cols, midpoint_compacted = self._rank_compress(midpoint_legalized)
        midpoint_compacted = self._place_ranked_grid_middle_bottom(
            midpoint_compacted,
            compact_rows,
            compact_cols,
            rows,
            cols,
        )
        if self._cells_are_available(midpoint_compacted, occupied_cells):
            return midpoint_compacted
        return self._point_embedding_nearest_free(midpoint_compacted, rows, cols, occupied_cells)

    def _cells_are_available(self, cells: list[tuple[int, int]], occupied_cells: set[tuple[int, int]]) -> bool:
        return len(set(cells)) == len(cells) and not any(cell in occupied_cells for cell in cells)

    def _rank_compress(self, points: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]]:
        '''Convert arbitrary points into a dense integer grid according to their relative location.
            Return resulting rows, columns, and location of each point.
            Example:
            [(10, 5), (2, 5), (10, 9), (7, 1)]
            -> (3, 3, [(2, 1), (0, 1), (2, 2), (1, 0)])'''
        if not points:
            return 0, 0, []

        distinct_x = sorted(set(point[0] for point in points))
        distinct_y = sorted(set(point[1] for point in points))

        x_rank = {x: rank for rank, x in enumerate(distinct_x)}
        y_rank = {y: rank for rank, y in enumerate(distinct_y)}

        ranked_points = [(x_rank[x], y_rank[y]) for x, y in points]
        num_rows = len(distinct_y)
        num_cols = len(distinct_x)

        return num_rows, num_cols, ranked_points

    def _affine_transform(
        self,
        points: list[tuple[int, int]],
        source_rows: int,
        source_cols: int,
        target_rows: int,
        target_cols: int,
    ) -> list[tuple[float, float]]:
        '''Map points in a source grid to ideal locations in a target grid.

            Input and output points use (x, y) = (column, row) order.
            The output coordinates are real-valued ideal positions. Collision
            resolution is handled by the later point-embedding step.
        '''
        if source_rows < 0 or source_cols < 0:
            raise ValueError("source grid dimensions must be non-negative")
        if target_rows <= 0 or target_cols <= 0:
            raise ValueError("target grid dimensions must be positive")
        if not points:
            return []

        if source_rows == 0 or source_cols == 0:
            raise ValueError("non-empty points require non-empty source grid dimensions")

        x_scale = 0.0 if source_cols == 1 else (target_cols - 1) / (source_cols - 1)
        y_scale = 0.0 if source_rows == 1 else (target_rows - 1) / (source_rows - 1)
        x_center = (target_cols - 1) / 2.0
        y_center = (target_rows - 1) / 2.0

        transformed = []
        for x, y in points:
            if not (0 <= x < source_cols and 0 <= y < source_rows):
                raise ValueError("point is outside the source grid")

            target_x = x_center if source_cols == 1 else x * x_scale
            target_y = y_center if source_rows == 1 else y * y_scale
            transformed.append((target_x, target_y))

        return transformed

    def _place_ranked_grid_middle_bottom(
        self,
        points: list[tuple[int, int]],
        source_rows: int,
        source_cols: int,
        target_rows: int,
        target_cols: int,
    ) -> list[tuple[int, int]]:
        if source_rows < 0 or source_cols < 0:
            raise ValueError("source grid dimensions must be non-negative")
        if target_rows <= 0 or target_cols <= 0:
            raise ValueError("target grid dimensions must be positive")
        if not points:
            return []
        if source_rows == 0 or source_cols == 0:
            raise ValueError("non-empty points require non-empty source grid dimensions")
        if source_rows > target_rows or source_cols > target_cols:
            raise ValueError("source grid does not fit into target grid")

        x_offset = (target_cols - source_cols + 1) // 2
        y_offset = 0
        placed = []
        for x, y in points:
            if not (0 <= x < source_cols and 0 <= y < source_rows):
                raise ValueError("point is outside the source grid")
            placed.append((x + x_offset, y + y_offset))
        return placed

    def _point_embedding_nearest_free(
        self,
        points: list[tuple[int, int]],
        rows: int,
        cols: int,
        occupied_cells = None,
    ):
        '''Assign ideal points to distinct grid cells by nearest-free repair.

            Input points are real-valued ideal locations in (x, y) =
            (column, row) order. The returned cells are integer (x, y)
            coordinates with 0 <= x < cols and 0 <= y < rows.
        '''
        if rows <= 0 or cols <= 0:
            raise ValueError("target grid dimensions must be positive")
        occupied_cells = set() if occupied_cells is None else set(occupied_cells)
        for cell in occupied_cells:
            if not (0 <= cell[0] < cols and 0 <= cell[1] < rows):
                raise ValueError("occupied cell is outside the target grid")
        if len(points) + len(occupied_cells) > rows * cols:
            raise ValueError("number of points exceeds available target grid capacity")
        if not points:
            return []

        groups = {}
        for index, point in enumerate(points):
            preferred = (
                self._round_to_grid(point[0], cols),
                self._round_to_grid(point[1], rows),
            )
            preferred_cost = self._squared_distance(preferred, point)
            groups.setdefault(preferred, []).append((preferred_cost, index, point))

        free = _FreeRows(rows, cols)
        for cell in occupied_cells:
            free.remove(cell)
        assignment = [None] * len(points)
        overflow = []

        for preferred, candidates in groups.items():
            candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
            if free.contains(preferred):
                _, winner_index, _ = candidates[0]
                assignment[winner_index] = preferred
                free.remove(preferred)
                overflow.extend(candidates[1:])
            else:
                overflow.extend(candidates)

        overflow.sort(key=lambda candidate: (candidate[0], candidate[1]))
        for _, index, point in overflow:
            assignment[index] = free.take_nearest(point)

        return assignment

    def _round_to_grid(self, value: float, size: int) -> int:
        rounded = int(math.floor(value + 0.5))
        return min(max(rounded, 0), size - 1)

    def _squared_distance(self, cell: tuple[int, int], point: tuple[float, float]) -> float:
        return (cell[0] - point[0]) ** 2 + (cell[1] - point[1]) ** 2


class _FenwickTree:
    def __init__(self, size: int):
        self.tree = [0] * (size + 1)

    def add(self, index: int, value: int):
        index += 1
        while index < len(self.tree):
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


class _FreeRows:
    def __init__(self, rows: int, cols: int):
        self.rows = [_IntervalSet(0, cols - 1) for _ in range(rows)]
        self.free_count = rows * cols

    def remove(self, cell: tuple[int, int]):
        col, row = cell
        self.rows[row].remove(col)
        self.free_count -= 1

    def contains(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        return self.rows[row].contains(col)

    def take_nearest(self, point: tuple[float, float]) -> tuple[int, int]:
        if self.free_count == 0:
            raise RuntimeError("no free cell found despite available grid capacity")

        best_cell = None
        best_key = None

        for row, free_cols in enumerate(self.rows):
            if free_cols.count == 0:
                continue

            vertical_cost = (row - point[1]) ** 2
            if best_key is not None and vertical_cost > best_key[0]:
                continue

            for col in _nearest_free_columns(free_cols, point[0]):
                key = (vertical_cost + (col - point[0]) ** 2, vertical_cost, row, col)
                if best_key is None or key < best_key:
                    best_cell = (col, row)
                    best_key = key

        if best_cell is None:
            raise RuntimeError("no free cell found despite available grid capacity")

        self.remove(best_cell)
        return best_cell


def _nearest_free_columns(free_cols, ideal_col: float) -> list[int]:
    columns = []

    left_count = free_cols.count_less_than(math.floor(ideal_col) + 1)
    if left_count > 0:
        columns.append(free_cols.kth(left_count - 1))

    right = free_cols.first_ge(math.ceil(ideal_col))
    if right is not None and right not in columns:
        columns.append(right)

    return columns


class _IntervalSet:
    def __init__(self, low: int, high: int):
        self.root = _IntervalNode(low, high) if low <= high else None

    @property
    def count(self) -> int:
        return _node_count(self.root)

    def first_ge(self, value: int):
        node = self.root
        result = None

        while node is not None:
            if value < node.start:
                result = node.start
                node = node.left
            elif value <= node.end:
                return value
            else:
                node = node.right

        return result

    def contains(self, value: int) -> bool:
        node = self.root
        while node is not None:
            if value < node.start:
                node = node.left
            elif value <= node.end:
                return True
            else:
                node = node.right
        return False

    def remove(self, value: int):
        self.root, removed = _remove(self.root, value)
        if not removed:
            raise KeyError(value)

    def count_less_than(self, value: int) -> int:
        return _count_less_than(self.root, value)

    def kth(self, index: int) -> int:
        if index < 0 or index >= self.count:
            raise IndexError("interval-set index out of range")

        node = self.root
        while node is not None:
            left_count = _node_count(node.left)
            if index < left_count:
                node = node.left
                continue

            index -= left_count
            interval_length = node.end - node.start + 1
            if index < interval_length:
                return node.start + index

            index -= interval_length
            node = node.right

        raise RuntimeError("interval-set subtree count is inconsistent")


class _IntervalNode:
    __slots__ = ("start", "end", "priority", "left", "right", "total")

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.priority = _priority(start, end)
        self.left = None
        self.right = None
        self.total = end - start + 1


def _remove(root, value: int):
    if root is None:
        return None, False

    if value < root.start:
        root.left, removed = _remove(root.left, value)
        return _update(root), removed

    if value > root.end:
        root.right, removed = _remove(root.right, value)
        return _update(root), removed

    if root.start == root.end:
        return _merge(root.left, root.right), True

    if value == root.start:
        root.start += 1
        return _update(root), True

    if value == root.end:
        root.end -= 1
        return _update(root), True

    old_end = root.end
    root.end = value - 1
    root = _update(root)
    root = _insert(root, _IntervalNode(value + 1, old_end))
    return root, True


def _insert(root, node):
    if root is None:
        return node

    if node.start < root.start:
        root.left = _insert(root.left, node)
        if root.left is not None and root.left.priority > root.priority:
            root = _rotate_right(root)
    elif node.start > root.start:
        root.right = _insert(root.right, node)
        if root.right is not None and root.right.priority > root.priority:
            root = _rotate_left(root)
    else:
        raise ValueError("duplicate interval start")

    return _update(root)


def _merge(left, right):
    if left is None:
        return right
    if right is None:
        return left

    if left.priority > right.priority:
        left.right = _merge(left.right, right)
        return _update(left)

    right.left = _merge(left, right.left)
    return _update(right)


def _rotate_left(root):
    new_root = root.right
    if new_root is None:
        return root

    root.right = new_root.left
    new_root.left = root
    _update(root)
    return _update(new_root)


def _rotate_right(root):
    new_root = root.left
    if new_root is None:
        return root

    root.left = new_root.right
    new_root.right = root
    _update(root)
    return _update(new_root)


def _count_less_than(root, value: int) -> int:
    if root is None:
        return 0

    if value <= root.start:
        return _count_less_than(root.left, value)

    if value > root.end:
        interval_length = root.end - root.start + 1
        return _node_count(root.left) + interval_length + _count_less_than(root.right, value)

    return _node_count(root.left) + (value - root.start)


def _update(node):
    node.total = _node_count(node.left) + (node.end - node.start + 1) + _node_count(node.right)
    return node


def _node_count(node) -> int:
    return node.total if node is not None else 0


def _priority(start: int, end: int) -> int:
    mask = (1 << 64) - 1
    value = (start * 0x9E3779B97F4A7C15 + end * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)
