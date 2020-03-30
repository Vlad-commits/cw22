import numpy as np
from sortedcollections import SortedListWithKey


def kstest(sample, cdf, test_points=None):
    import time
    start_time = time.time()

    first_point = sample[0]
    if not (hasattr(first_point, "__len__")):
        dim = 1
    else:
        dim = len(first_point)
    e_cdf_index = ECDFIndex(dim)

    real_cdf_values = []
    result = []
    for index, value in enumerate(sample):
        e_cdf_index.add(value)
        real_cdf_values.append(cdf(value))
        print("index: " + str(index) + " time spend: " + str(time.time() - start_time))
        if test_points is None or index in test_points:
            ecdf_vals = e_cdf_index.edf_values()
            Dplus = np.abs((ecdf_vals - real_cdf_values)).max()
            Dmin = np.abs((np.subtract(real_cdf_values, (ecdf_vals - 1 / (index + 1))))).max()
            D = np.max([Dplus, Dmin])
            result.append(D)
    return result


class IndexedPoint:
    def __init__(self,
                 point,
                 point_number: int):
        self.point_number = point_number
        self.point = point


class SingleDimensionIndex:

    def __init__(self, key_extractor):
        self.next_point_number = 0
        self.key_extractor = key_extractor
        self.index = SortedListWithKey(key=lambda indexed_point: key_extractor(indexed_point.point))
        self.inverted_index = {}

    def add(self, point):
        indexed_point = IndexedPoint(point, self.next_point_number)
        pos = self.index.bisect_key_left(self.key_extractor(point))
        self.index.add(indexed_point)
        self.inverted_index[indexed_point.point_number] = set([p.point_number for p in self.index[:pos]])
        for point_to_update in self.index[pos:]:
            self.inverted_index[point_to_update.point_number].add(indexed_point.point_number)
        self.next_point_number += 1

    def get_point_numbers_lte(self):
        return self.inverted_index


class ECDFIndex:

    def __init__(self, dim: int):
        self.dim = dim
        if dim != 1:
            self.indices = [SingleDimensionIndex(lambda point: point[i]) for i in range(dim)]
        else:
            self.indices = [SingleDimensionIndex(lambda point: point)]
        self.points = []

    def add(self, point):
        self.points.append(point)
        for index in self.indices:
            index.add(point)

    def edf_values(self):
        points_size = len(self.points)
        merged = [None] * points_size
        for index in self.indices:
            sorted = index.get_point_numbers_lte()
            for point_number in sorted.keys():
                points_lte_it = sorted[point_number]

                current = merged[point_number]
                if current is None:
                    merged[point_number] = points_lte_it
                else:
                    merged[point_number] = {pn for pn in current
                                            if pn in points_lte_it}

        return np.array([len(points_lte_ith) / points_size for points_lte_ith in merged])
