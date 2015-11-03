import random
import math
import numpy

__author__ = 'ay27'

import unittest
import NMF
import nmf1
import csv


def random_mask(R, n, m, count=20):
    ret_r = R.copy()
    for i in range(count):
        rand_n = random.randint(0, n - 1)
        rand_m = random.randint(0, m - 1)
        ret_r[rand_n][rand_m] = 0.0
    return ret_r


def calc_mat(X, Y):
    e = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            e += pow(X[i][j] - Y[i][j], 2)
    return math.sqrt(e)


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
        test_r = random_mask(self.R, self.N, self.M)
        k = 6
        P = numpy.random.rand(self.N, k)
        Q = numpy.random.rand(self.M, k)

        fr = open('log', mode='w')

        nP, nQ = NMF.nmf_gd(test_r, P, Q, k, fr, steps=2000)
        nR = numpy.dot(nP, nQ.T)
        fr.write('\nR:\n')
        fr.write(self.R)
        fr.write('\nresult_R:\n')
        fr.write(nR)
        fr.write('\nP:\n')
        fr.write(nP)
        fr.write('\nQ:\n')
        fr.write(nQ)
        fr.close()

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

    def test_2(self):
        data = []
        with open('out.txt') as f:
            for line in f.readlines():
                if line.startswith('e = '):
                    data.append(float(line.split()[2]))
        print(data)


if __name__ == '__main__':
    unittest.main()
