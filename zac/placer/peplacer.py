from zac.ds.architecture import Architecture
import numpy as np
import math
from copy import deepcopy
import gc

class PointEmbeddingPlacer:
    """class to find a qubit layout via point-grid embedding."""

    def __init__(self):
        pass

    # return gate mapping of one layer
    # architecture.entanglement_zone is a list of SLM id list. shape: [[id1, id2]]
    # use these id to find SLM objects in architecture.dict_SLM 
    # list_gate: gates in one layer, each gate is a 2-tuple
    # storage_mapping[i] is the location of the ith qubit, in the format of (slm_id, y, x)
    def run(self, architecture: Architecture, list_gate: [[int, int]], storage_mapping: [(int, int, int)]) -> (list, list):
        # get entanglement zone width and height
        slmid_l, slmid_r = architecture.entanglement_zone[0]
        slm0 = architecture.dict_SLM[slmid_l]
        ezh, ezw = slm0.n_r, slm0.n_c

        K = len(list_gate)
        assert ezh * ezw >= K
        
        # extract gate midpoints
        midpoints = []
        for q1, q2 in list_gate:
            loc1, loc2 = storage_mapping[q1], storage_mapping[q2]
            # TODO: now assume all qubits are in the storage zone. Handle reuse in the future.
            midy = loc1[1] + loc2[1] # no need to divide by 2, since we only need the relative location
            midx = loc1[2] + loc2[2]
            midpoints.append((midx, midy))

        # rank-compress
        rows, cols, midpoints_ranked = self._rank_compress(midpoints)

        # affine transform
        # TODO: if rank compressed grid fits into entanglement zone, no affine transform is needed
        midpoint_transformed = self._affine_transform(midpoints_ranked, rows, cols, ezh, ezw)

        # point embedding
        midpoint_legalized = self._point_embedding_nearest_free(midpoint_transformed, ezh, ezw)

        # full mapping generation
        new_mapping = deepcopy(storage_mapping)
        for i, (x, y) in enumerate(midpoint_legalized):
            q1, q2 = list_gate[i]
            q1_x = storage_mapping[q1][2]
            q2_x = storage_mapping[q2][2]
            # keep the original left-right relationship of each qubit pair
            if q1_x > q2_x:
                q1, q2 = q2, q1
            new_mapping[q1] = (slmid_l, y, x)
            new_mapping[q2] = (slmid_r, y, x)
        return new_mapping

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

    def _point_embedding_nearest_free(self, points: list[tuple[int, int]], rows: int, cols: int):
        '''Assign ideal points to distinct grid cells by nearest-free repair.

            Input points are real-valued ideal locations in (x, y) =
            (column, row) order. The returned cells are integer (x, y)
            coordinates with 0 <= x < cols and 0 <= y < rows.
        '''
        if rows <= 0 or cols <= 0:
            raise ValueError("target grid dimensions must be positive")
        if len(points) > rows * cols:
            raise ValueError("number of points exceeds target grid capacity")
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
        assignment = [None] * len(points)
        overflow = []

        for preferred, candidates in groups.items():
            candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
            _, winner_index, _ = candidates[0]
            assignment[winner_index] = preferred
            free.remove(preferred)
            overflow.extend(candidates[1:])

        overflow.sort(key=lambda candidate: (candidate[0], candidate[1]))
        for _, index, point in overflow:
            assignment[index] = free.take_nearest(point)

        return assignment

    def _round_to_grid(self, value: float, size: int) -> int:
        rounded = int(math.floor(value + 0.5))
        return min(max(rounded, 0), size - 1)

    def _squared_distance(self, cell: tuple[int, int], point: tuple[float, float]) -> float:
        return (cell[0] - point[0]) ** 2 + (cell[1] - point[1]) ** 2


class _FreeRows:
    def __init__(self, rows: int, cols: int):
        self.rows = [_IntervalSet(0, cols - 1) for _ in range(rows)]
        self.free_count = rows * cols

    def remove(self, cell: tuple[int, int]):
        col, row = cell
        self.rows[row].remove(col)
        self.free_count -= 1

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
                key = (vertical_cost + (col - point[0]) ** 2, row, col)
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
