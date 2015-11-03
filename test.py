import random
import numpy

__author__ = 'ay27'

import unittest
import NMF
import nmf1
from sklearn import decomposition
import csv


def random_mask(R, n, m, count=20):
    ret_r = R.copy()
    for i in range(count):
        rand_n = random.randint(0, n-1)
        rand_m = random.randint(0, m-1)
        ret_r[rand_n][rand_m] = 0.0
    return ret_r


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # R = [
        #     [5, 3, 0, 1],
        #     [4, 0, 0, 1],
        #     [1, 1, 0, 5],
        #     [1, 0, 0, 4],
        #     [0, 1, 5, 4],
        # ]
        # self.R = numpy.array(R)
        #
        # self.N = len(R)
        # self.M = len(R[0])
        # self.K = 2
        #
        # self.P = numpy.random.rand(self.N, self.K)
        # self.Q = numpy.random.rand(self.M, self.K)
        self.R = read_from_csv('emotions-train.csv')
        self.N = len(self.R)
        self.M = len(self.R[0])

    def test_1(self):
        print(self.R)
        # test_r = random_mask(self.R, self.N, self.M)
        # k = 20
        # P = numpy.random.rand(self.N, k)
        # Q = numpy.random.rand(self.M, k)
        # nP, nQ = NMF.nmf_gd(self.R, P, Q, k)
        # nR = numpy.dot(nP, nQ.T)
        # print(calc_mat(self.R, nR))

        # read_from_csv('/Users/ay27/PycharmProjects/NMF/dataset/medical/medical-test.csv')
        # read_from_csv('/Users/ay27/PycharmProjects/NMF/dataset/emotions/emotions-train.csv')

        # def test_1(self):
        #     nP, nQ = NMF.nmf_gd(self.R, self.P, self.Q, self.K)
        #     nR = numpy.dot(nP, nQ.T)
        #
        #     print('P\n')
        #     print(nP)
        #     print('Q\n')
        #     print(nQ)
        #     print('R\n')
        #     print(nR)
        #
        # def test_2(self):
        #     nP, nQ = NMF.nmf_mul(self.P, self.Q.T, self.R)
        #     nR = numpy.dot(nP, nQ.T)
        #
        #     print('P\n')
        #     print(nP)
        #     print('Q\n')
        #     print(nQ)
        #     print('R\n')
        #     print(nR)
        #
        # def test_3(self):
        #     W, H = nmf1.nmf(self.R, self.P, self.Q.T, 0.000001, 1000000000, 50000)
        #     print('W\n', W)
        #     print('\nH\n', H)
        #     print('\nWH\n', numpy.dot(W, H))
        #
        # def test_4(self):
        #     model = decomposition.ProjectedGradientNMF(2, init='random')
        #     nmf = model.fit(numpy.array(self.R))
        #     print(nmf.components_)


if __name__ == '__main__':
    unittest.main()


def calc_mat(X, Y):
    e = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            e += pow(X[i][j] - Y[i][j], 2)
    return e


def read_from_csv(filename):
    R = []
    with open(filename) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            tmp = []
            for i in range(72):
                tmp.append(float(row[i]))
            R.append(tmp)
    return R