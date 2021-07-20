import cv2
import numpy as np
from collections import OrderedDict
import math
import copy
import functools
import time


def clock(func):
    """this is outer clock function"""

    @functools.wraps(func)  # --> 4
    def clocked(*args, **kwargs):  # -- 1
        """this is inner clocked function"""
        start_time = time.time()
        result = func(*args, **kwargs)  # --> 2
        time_cost = time.time() - start_time
        print(func.__name__ + " func time_cost -> {:f}".format(time_cost))
        return result

    return clocked  # --> 3


def parse_standardInf(stdInf):
    """
    version2:
    add key word: side
    """
    ret_data = []
    for std_point in stdInf:
        if std_point.get('groupType', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            tmp = [float(std_point['pointX']),
                   float(std_point['pointY']),
                   float(std_point['pointZ']),
                   str(std_point['groupType']),
                   int(std_point['pointId'])-404]

        elif std_point.get('insertsType', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            tmp = [float(std_point['pointX']),
                   float(std_point['pointY']),
                   float(std_point['pointZ']),
                   str(std_point['insertsType']),
                   int(std_point['pointId'])]
        else:
            continue

        if std_point.get('side', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            tmp.append(str(std_point['side']))

        ret_data.append(tmp)
    return ret_data


def parse_predictionInf(testInf: list):
    """
    version2:
    add key words: side, position
    """
    ret_data = []
    for test_point in testInf:
        # 避免testInfo 为空，报错
        if test_point.get('worldX', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            tmp = [float(test_point['worldX']),
                   float(test_point['worldY']),
                   float(test_point['worldZ']),
                   str(test_point['label']),
                   float(test_point['confidence'])]
        if test_point.get('side', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            tmp.append(str(test_point['side']))
            tmp.append(str(test_point['position']))
        ret_data.append(tmp)
    ret_data = sorted(ret_data, key=lambda x: -x[4])
    return ret_data


# @clock
def calculate_mutual_distance(pointSet):
    """
    :param pointSet: points set info [x,y,z]
    calculate the mutual distance between std points
    calculate the mutual distance between test points
    :return: distance matrix:numpy.array[len*len]
    """
    __len = len(pointSet)
    ret_mat = np.zeros((__len, __len), dtype=float)

    for i, _ in enumerate(pointSet):
        __tiled_mat = np.tile(pointSet[i], (__len, 1))
        __dis = pointSet - __tiled_mat
        __dis = np.multiply(__dis, __dis)
        __dis = np.sum(__dis, axis=1)
        __dis = np.sqrt(__dis)
        ret_mat[i] = __dis.reshape(-1)
    return ret_mat


@clock
def calculate_mutual_angle(points, distances):
    __len = len(points)
    __mutual_angle = np.zeros((__len, __len, __len), dtype=float)
    for i in range(__len):
        for j in range(__len):
            if i==j:
                continue
            vec_j2i = points[i] - points[j]
            vec_j2i = np.tile(vec_j2i, (len(points), 1))
            vec_j2other = points - np.tile(points[j], (len(points), 1))
            l2_vec_j2i = np.linalg.norm(vec_j2i,axis=1)
            l2_vec_j2other = np.linalg.norm(vec_j2other,axis=1)
            dot = np.dot(vec_j2i,vec_j2other.T)
            dot = vec_j2i* vec_j2other
            dot = np.sum(dot, axis=1)
            cos = dot / (l2_vec_j2i * l2_vec_j2other)
            __mutual_angle[i][j] = np.arccos(cos)
    return __mutual_angle


def create_feature_matrix(distanceMat, topK):
    __len = len(distanceMat)
    ret_feature_mat = np.zeros((__len, topK), dtype=float)
    for i in range(__len):
        sorted_index = np.argsort(distanceMat[i])
        ret_feature_mat[i] = distanceMat[i][sorted_index][1:topK + 1]
    return ret_feature_mat


def calculate_angle_by_acos_C(a, b, c):
    return math.acos((a * a + b * b - c * c) / (2 * a * b))


def draw_point_in_code(points, name):
    # points = np.dtype(np.uint8)
    points = np.array(points, dtype=np.int16)
    mat = np.ones((2000, 1000, 3)) * 255
    for i, p in enumerate(points):
        x, y = int(p[0] + 800), int(p[1] + 800)
        print(f'{i}, x,y:{x,y}')
        if x==424+800 and y==78+800:
            mat = cv2.putText(mat, str(i), (x//2, y//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            mat = cv2.putText(mat, str(i), (x // 2, y // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow(name, mat)
    cv2.waitKey(0)
    return


class subClass_matchPoint:
    '''
    说明
    '''
    # @clock
    def __init__(self, stdInf, debug=False):
        self.__debug = debug
        if self.__debug:
            from Logger import Logger
            import logging
            self.__logger = Logger(log_file_name='../log.txt', log_level=logging.DEBUG,
                                   logger_name="point_match_logger").get_log()
        if len(stdInf) < 1 and self.__debug:
            self.__logger.info('standard workpiece is none')
        self.__stdInfAll = parse_standardInf(stdInf)
        self.__workpieceName2id = dict()
        self.__stdInf = self.__preprocess(self.__stdInfAll)
        self.__std2predDict = OrderedDict()
        self.__std2predDict4symmetry = OrderedDict()
        if len(stdInf) > 0:
            self.__heightThreshold = 10
            self.__distanceThreshold = 8
            self.__angleThreshold = 5
            self.__ransacThreshold = 10
            self.__l1_l2_regThreshold = 8
        # final return value
        self.__ret = []
        return

    # @clock
    def __preprocess(self, pointsInf):
        """
        since the type of workpiece is string, so before covert the points info into np.array,
        it should be transfer to index.
        :param pointsInf:
        :return: processed points info [and workpiece-name-to-index dictionary]
        """
        ret = []
        for i, pointInfo in enumerate(pointsInf):
            # add new workpiece name to the dict
            pointInfo = pointInfo[:5]
            if pointInfo[3] not in self.__workpieceName2id.keys():
                self.__workpieceName2id.update({pointInfo[3]: len(self.__workpieceName2id) + 1})
            # pointInfo[i][3] = self.__workpieceName2id[pointInfo[3]]
            pointInfo[3] = self.__workpieceName2id[pointInfo[3]]
            ret.append(pointInfo)
        return np.array(ret)

    # @clock
    def __init_after_get_pred_points(self, predInf):
        if len(predInf) < 1 and self.__debug:
            self.__logger.info('predict workpiece is none')
        self.__predInfAll = parse_predictionInf(predInf)
        self.__predInf = self.__preprocess(self.__predInfAll)
        if len(self.__stdInf) > 1 and len(self.__predInf) > 1:
            self.__stdMutualDistance = calculate_mutual_distance(self.__stdInf)
            self.__predMutualDistance = calculate_mutual_distance(self.__predInf)
            if len(self.__stdInf) > 2 and len(self.__predInf) > 2:
                # stdProcess = Thread(target=calculate_mutual_angle, args=(self.__stdInf[:, :3], self.__stdMutualDistance))
                # predProcess = Thread(target=calculate_mutual_angle, args=(self.__predInf[:, :3], self.__predMutualDistance))
                # stdProcess.start()
                # predProcess.start()
                # stdProcess.join()
                # predProcess.join()
                # self.__stdMutualAngle = calculate_mutual_angle(self.__stdInf[:, :3], self.__stdMutualDistance)
                # self.__predMutualAngle = calculate_mutual_angle(self.__predInf[:, :3], self.__predMutualDistance)
                self.__topK = min(3, len(self.__predInf) - 1, len(self.__stdInf) - 1)
                self.__stdFeatureMatrix = create_feature_matrix(self.__stdMutualDistance, self.__topK)
                self.__predFeatureMatrix = create_feature_matrix(self.__predMutualDistance, self.__topK)
                self.__reflectionPairs = list()
                self.__rotationalSymmetry = False
                self.__reflection_search()
        if self.__debug:
            self.__logger.info('self.__workpieceName2id: \n%s' % str(self.__workpieceName2id))
            self.__logger.info('self.__stdInf: \n%s' % str(self.__stdInf))
            self.__logger.info('self.__predInf\n%s' % str(self.__predInf))
            if len(self.__stdInf) > 1 and len(self.__predInf) > 1:
                self.__logger.info('self.__stdMutualDistance\n%s' % str(self.__stdMutualDistance))
                self.__logger.info('self.__predMutualDistance\n%s' % str(self.__predMutualDistance))
                if len(self.__stdInf) > 2 and len(self.__predInf) > 2:
                    # self.__logger.info('self.__stdMutualAngle\n%s'% str(self.__stdMutualAngle))
                    # self.__logger.info('self.__predMutualAngle\n%s'% str(self.__predMutualAngle))
                    self.__logger.info('self.__stdFeatureMatrix\n%s' % str(self.__stdFeatureMatrix))
                    self.__logger.info('self.__predFeatureMatrix\n%s' % str(self.__predFeatureMatrix))
                    self.__logger.info('self.__reflectionPairs\n%s' % str(self.__reflectionPairs))
            if self.__debug:
                dstPoints = np.float32(self.__predInf[:, :3])
                srcPoints = np.float32(self.__stdInf[:, :3])
                draw_point_in_code(dstPoints, 'test')
                draw_point_in_code(srcPoints, 'std')

    def __reflection_search(self):
        '''
        寻找轴对称的点 pair
        :return:
        '''
        if len(self.__stdInf) > 1 and len(self.__predInf) > 1:
            # self.__stdMutualDistance 已存在
            __stdMutualDistance = copy.deepcopy(self.__stdMutualDistance)
            __stdMutualDistance.sort(axis=1)
            for i, distance_feature in enumerate(__stdMutualDistance):
                # height filter
                __filtered_index = self.__height_filter(self.__stdInf[i][2])
                # label filter
                __filtered_index = filter(lambda x: self.__stdInf[x][3] == self.__stdInf[i][3], __filtered_index)
                __filtered_index = np.array(list(__filtered_index))
                if 0 == len(__filtered_index):
                    continue
                tiled_mat = np.tile(distance_feature, (len(__filtered_index), 1))
                distance_err = __stdMutualDistance[__filtered_index] - tiled_mat
                distance_err = np.multiply(distance_err, distance_err)
                distance_err = np.sum(distance_err, axis=1)
                distance_err = distance_err / (len(__stdMutualDistance) - 1)
                distance_err = np.sqrt(distance_err)
                distance_err = distance_err.reshape(-1)
                index_pair = np.where(distance_err < 3)[0].tolist()
                # 轴对称
                if len(index_pair) == 2 and index_pair not in self.__reflectionPairs:
                    self.__reflectionPairs.append(index_pair)
                # 旋转对称
                elif len(index_pair) > 2:
                    self.__rotationalSymmetry = True
                    break
        return

    def __ans_format_covert(self):
        new_dict = {v: k for k, v in self.__workpieceName2id.items()}
        for i, testPoint in enumerate(self.__predInf):
            pointDict = dict({"pointId": -1,
                              "worldX": testPoint[0],
                              "worldY": testPoint[1],
                              "worldZ": testPoint[2],
                              "label": str(new_dict[testPoint[3]])})
            if len(self.__predInfAll[i])>5:
                pointDict['side'] = str(self.__predInfAll[i][5])
                pointDict['position'] =str(self.__predInfAll[i][6])

            if i in self.__std2predDict.values():
                std2test = {v: k for k, v in self.__std2predDict.items()}
                key = std2test.get(i, None)
                if key is not None:
                    pointDict["pointId"] = int(self.__stdInf[key][4])
            self.__ret.append(pointDict)

    def match_point(self, predPoints):
        self.__init_after_get_pred_points(predPoints)
        switch = {0: self.__case0, 1: self.__case1, 2: self.__case2, 3: self.__case3}
        case_index = min(len(self.__predInf), 3)
        if self.__debug:
            self.__logger.info('case %s' % str(case_index))
        switch.get(case_index)()
        # print(self.__std2predDict)
        # exit()
        self.__ans_format_covert()
        return self.__ret

    # 根据std points 数量来分情况
    def __case0(self):
        return

    def __case1(self):
        if len(self.__stdInf) < 1:
            if self.__debug:
                self.__logger.info('there is no point in standard point set!')
            return
        else:
            self.__find_first_pair()

    def __case2(self):
        if len(self.__stdInf) < 1:
            return
        elif 1 == len(self.__stdInf):
            self.__find_first_pair()
        elif len(self.__stdInf) > 1:
            self.__find_first_pair()
            self.__find_rest_pair()

    def __find_first_pair(self):
        if self.__debug:
            self.__logger.info('START: find first pair!')
        for i, pred_insert in enumerate(self.__predInf):
            if self.__debug:
                self.__logger.info('prediction candidate index: %s, %s' % (str(i), str(pred_insert)))
            __filtered_index = self.__height_filter(pred_insert[2])
            if self.__debug:
                self.__logger.info('height filter: %s' % str(__filtered_index))
            if 0 == len(__filtered_index):
                continue
            __filtered_index = self.__label_filter(__filtered_index, i)
            if self.__debug:
                self.__logger.info('label filter: %s' % str(__filtered_index))
            if 0 == len(__filtered_index):
                continue
            if len(self.__std2predDict)>0:
                __filtered_index = self.__distance_filter(__filtered_index, i)
                if self.__debug:
                    self.__logger.info('distance filter: %s' % str(__filtered_index))
                if 0 == len(__filtered_index):
                    continue
            if len(self.__stdInf) > 2 and len(self.__predInf) > 2:
                __filtered_index_by_feature_vector = self.__feature_vector_filter(
                    self.__stdFeatureMatrix[__filtered_index],
                    self.__predFeatureMatrix[i])
                if 0 == len(__filtered_index_by_feature_vector):
                    continue
                __filtered_index = __filtered_index[__filtered_index_by_feature_vector]
                if self.__debug:
                    self.__logger.info('feature vector filter: %s' % str(__filtered_index))
            if 1 == len(__filtered_index):
                self.__std2predDict.update({__filtered_index[0]: i})
                if self.__debug:
                    self.__logger.info('add pair %s' % str({__filtered_index[0]: i}))
            if 2 == len(__filtered_index) and __filtered_index.tolist() in self.__reflectionPairs:
                self.__std2predDict.update({__filtered_index[0]: i})
                self.__std2predDict4symmetry.update({__filtered_index[1]: i})
                if self.__debug:
                    self.__logger.info('add pair to __std2predDict: %s' % str({__filtered_index[0]: i}))
                    self.__logger.info('add pair to __std2predDict4symmetry: %s' % str({__filtered_index[1]: i}))
                break

    def __height_filter(self, height):
        __stdInf_height = self.__stdInf[:, 2].reshape(-1)
        __filtered_index = np.where((__stdInf_height > height - self.__heightThreshold) & (
                __stdInf_height < height + self.__heightThreshold))
        __filtered_index = filter(lambda x: x not in self.__std2predDict.keys(), __filtered_index[0])
        return np.array(list(__filtered_index))

    def __label_filter(self, std_index_array, pred_index):
        ans = filter(lambda x: self.__stdInf[x][3] == self.__predInf[pred_index][3], std_index_array)
        return np.array(list(ans))

    def __feature_vector_filter(self, stdFeatureMatrix, predFeatureVec):
        tiled_mat = np.tile(predFeatureVec, (len(stdFeatureMatrix), 1))
        err = stdFeatureMatrix[:, :3] - tiled_mat
        # l1 distance
        # err = np.abs(err)
        # err = np.sum(err, axis=1)
        # err = err / 3

        # l2 distance
        err = np.multiply(err, err)
        err = np.sum(err, axis=1)
        err = err / self.__topK
        err = np.sqrt(err)

        err = err.reshape(-1)
        if self.__debug:
            # self.__logger.info('add pair %s' % str({__filtered_index[0]: i}))
            self.__logger.info('feature error mat: %s' % str(err))
        candidate_index_set = np.where(err <= self.__l1_l2_regThreshold)
        # candidate_index_set = filter(lambda x: x not in self.__std2predDict.keys(), candidate_index_set[0])
        return np.array(list(candidate_index_set[0]))

    def __distance_filter(self, std_index_array, pred_index):
        ans = np.zeros_like(std_index_array)
        for i, std_index in enumerate(std_index_array):
            flag = True
            for key, val in self.__std2predDict.items():
                # for reflection
                # if key in self.__reflectionPairs.reshape(-1):
                # if key in [item for pair in self.__reflectionPairs for item in pair]:
                #     continue
                if not self.__meet_distance_condition(self.__stdMutualDistance[std_index][key],
                                                      self.__predMutualDistance[pred_index][val]):
                    flag = False
            if flag:
                ans[i] = 1
        __ans_index = np.where(ans == 1)
        return np.array(list(std_index_array[__ans_index]))

    def __meet_distance_condition(self, dis1, dis2):
        return abs(dis1 - dis2) <= self.__distanceThreshold

    def __calculate_angle_scope(self, side_length):
        cos_radians = math.acos((2 * side_length * side_length - self.__angleThreshold * self.__angleThreshold) / (
                2 * side_length * side_length))
        return cos_radians

    # 第一版，只匹配前两个点，因为多了，反而没用
    # def __meet_angle_condition(self):
    def __angle_filter(self, std_index_array, pred_index):
        keys, values = list(self.__std2predDict.keys()), list(self.__std2predDict.values())
        ans = np.zeros_like(std_index_array)
        for i, std_index in enumerate(std_index_array):
            error_scope1 = self.__calculate_angle_scope(self.__stdMutualDistance[keys[0]][std_index])
            error_scope2 = self.__calculate_angle_scope(self.__stdMutualDistance[keys[1]][std_index])
            test_angle = calculate_angle_by_acos_C(self.__predMutualDistance[pred_index][values[0]],
                                                   self.__predMutualDistance[pred_index][values[1]],
                                                   self.__predMutualDistance[values[0]][values[1]])
            std_angle = calculate_angle_by_acos_C(self.__stdMutualDistance[std_index][keys[0]],
                                                  self.__stdMutualDistance[std_index][keys[1]],
                                                  self.__stdMutualDistance[keys[0]][keys[1]], )
            if std_angle - error_scope1 - error_scope2 <= test_angle <= std_angle + error_scope1 + error_scope2:
                ans[i] = 1
        __ans_index = np.where(ans == 1)
        return np.array(list(std_index_array[__ans_index]))

    def __find_rest_pair(self):
        if self.__debug:
            self.__logger.info('NEXT: find rest pair!')
        for i, pred_insert in enumerate(self.__predInf):
            if self.__debug:
                self.__logger.info('prediction candidate index: %s, %s' % (str(i), str(pred_insert)))
            if i in self.__std2predDict.values():
                if self.__debug:
                    self.__logger.info('%s has been in dict, continue' % str(i))
                continue
            # height filter
            __filtered_index = self.__height_filter(pred_insert[2])
            if self.__debug:
                self.__logger.info('height filter: %s' % str(__filtered_index))
            if 0 == len(__filtered_index):
                continue
            # label filter
            __filtered_index = self.__label_filter(__filtered_index, i)
            if self.__debug:
                self.__logger.info('label filter: %s' % str(__filtered_index))
            if 0 == len(__filtered_index):
                continue
            # distance filter 此时已经找到第一个匹配点
            __filtered_index = self.__distance_filter(__filtered_index, i)
            if self.__debug:
                self.__logger.info('distance filter: %s' % str(__filtered_index))
            if 0 == len(__filtered_index):
                continue
            # feature vector filter
            if len(self.__stdInf) > 2 and len(self.__predInf) > 2:
                __filtered_index_by_feature_vector = self.__feature_vector_filter(
                    self.__stdFeatureMatrix[__filtered_index],
                    self.__predFeatureMatrix[i])
                if self.__debug:
                    self.__logger.info(
                        '__filtered_index_by_feature_vector filter: %s' % str(__filtered_index_by_feature_vector))
                if 0 == len(__filtered_index_by_feature_vector):
                    continue
                __filtered_index = __filtered_index[__filtered_index_by_feature_vector]

                if self.__debug:
                    self.__logger.info('feature vector filter: %s' % str(__filtered_index))
                if 0 == len(__filtered_index):
                    continue
            # angle filter
            if len(self.__std2predDict) > 1:
                __filtered_index = self.__angle_filter(__filtered_index, i)
                if self.__debug:
                    self.__logger.info('angle filter: %s' % str(__filtered_index))
                if 0 == len(__filtered_index):
                    continue
            if 1 == len(__filtered_index):
                if self.__debug:
                    self.__logger.info('add pair %s' % str({__filtered_index[0]: i}))
                self.__std2predDict.update({__filtered_index[0]: i})
            # if 2 == len(__filtered_index) and len(self.__std2predDict) > 1:
            #     __filtered_index = __filtered_index.tolist().sort()
            #     if __filtered_index in self.__reflectionPairs:
            #         if self.__debug:
            #             self.__logger.info('add reflection pair %s' % str({__filtered_index[0]: i}))
            #             self.__logger.info('add reflection pair %s' % str({__filtered_index[1]: i}))
            #         self.__std2predDict.update({__filtered_index[0]: i})
            #         self.__std2predDict.update({__filtered_index[1]: i})

    def __calculate_transformed_pred_point_by_transMat(self, transMat):
        a = [0, 0, 0]
        b = [0, 0, 0, 1]
        # 3*3 -> 4*4
        offsetMat = np.zeros(4)
        offsetMat[:2] = transMat[:2, 2]

        rotateMat = np.insert(transMat, transMat.shape[0], values=a, axis=0)
        rotateMat = np.insert(rotateMat, rotateMat.shape[1], values=b, axis=1)
        rotateMat[:3, 2] = [0, 0, 1]

        pred_point_set = self.__predInf[:, :3]
        c = [1 for _ in range(len(self.__predInf))]
        pred_point_set = np.insert(pred_point_set, pred_point_set.shape[1], values=c, axis=1)
        transformed_pred_point_set = []
        for pred_point in pred_point_set:
            pred_point = pred_point.reshape(-1, 1)
            transformed_pred_point = np.matmul(rotateMat, pred_point)
            transformed_pred_point = transformed_pred_point.reshape(-1)
            transformed_pred_point = transformed_pred_point + offsetMat
            transformed_pred_point_set.append(transformed_pred_point[:3])
        return np.array(transformed_pred_point_set)

    def __calculate_distance_for_transformed_and_std_points(self, transformed_pred_point_set):
        '''
        :param transformed_test_point_set:
        :return: 一个n*m的 distance matrix：
        其中，n为transformed_test_point_set数量，
        m为self.__stdInfo 数量
        distance[i][j]表示 第i个test点到第j个std点的距离
        '''
        std_point_set = self.__stdInf[:, :3]
        distance_matrix = np.zeros((transformed_pred_point_set.shape[0], std_point_set.shape[0]))
        for i, p in enumerate(transformed_pred_point_set):
            p = np.repeat(p, std_point_set.shape[0], axis=0).reshape(3, std_point_set.shape[0]).T
            __tmp = np.sqrt(np.sum((p - std_point_set) ** 2, axis=1))
            for stdIndex, dis in enumerate(__tmp):
                if self.__stdInf[stdIndex][3] != self.__predInf[i][3]:
                    __tmp[stdIndex] = np.inf
            distance_matrix[i] = __tmp
        return distance_matrix

    # @clock
    def __match_point_by_transMat_and_mask(self, transMat, mask):
        std2predDict = copy.deepcopy(self.__std2predDict)
        mask = mask.reshape(-1)
        if self.__debug:
            self.__logger.info('mask %s' % str(mask))
        self.__std2predDict.clear()
        for i, key_val in enumerate(std2predDict.items()):
            if mask[i]:
                key, val = key_val
                self.__std2predDict.update({key: val})
            else:
                if self.__debug:
                    key, val = key_val
                    self.__logger.info(f'del pair {key}: {val}')
        # calculate the testInfo after transformed
        transformed_pred_point_set = self.__calculate_transformed_pred_point_by_transMat(transMat)

        distance_matrix = self.__calculate_distance_for_transformed_and_std_points(transformed_pred_point_set)
        ## TODO: <50 并且数量为1，添加，   否则 计算feature
        min_distance_index_per_row = np.where(
            (distance_matrix == np.min(distance_matrix, axis=0)) & (distance_matrix < 50))
        # (distance_matrix == np.min(distance_matrix, axis=0)))
        __test_index, __std_index = min_distance_index_per_row
        for i in range(len(__std_index)):
            if __std_index[i] not in self.__std2predDict.keys() and __test_index[i] not in self.__std2predDict.values():
                if self.__debug:
                    self.__logger.info('add pair %s' % str({__std_index[i]: __test_index[i]}))
                self.__std2predDict.update({__std_index[i]: __test_index[i]})

    # @clock
    def __case3(self):
        if len(self.__stdInf) < 1:
            return
        elif 1 == len(self.__stdInf):
            self.__find_first_pair()
        elif len(self.__stdInf) > 1:
            self.__find_first_pair()
            if len(self.__std2predDict) > 0:
                self.__find_rest_pair()
                self.__find_rest_pair()
        # 此时已经得到pair 对
        if self.__debug:
            self.__logger.info('current dict %s' % str(self.__std2predDict))
        if len(self.__std2predDict) >= 3:
            dstPoints = np.float32(self.__stdInf[:, :2][list(self.__std2predDict.keys())])
            srcPoints = np.float32(self.__predInf[:, :2][list(self.__std2predDict.values())])
            # ransac
            # transformation_matrix, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, self.__ransacThreshold)

            # estimateAffinePartial2D
            transformation_matrix, mask = cv2.estimateAffinePartial2D(srcPoints[:, :2], dstPoints[:, :2])
            transformation_matrix = np.insert(transformation_matrix, 2, values=[0, 0, 1], axis=0)

            self.__match_point_by_transMat_and_mask(transformation_matrix, mask)

        answer1 = copy.deepcopy(self.__std2predDict)
        # if 有对称
        if len(self.__std2predDict4symmetry):
            self.__std2predDict = self.__std2predDict4symmetry
            if self.__debug:
                self.__logger.info('current dict2 %s' % str(self.__std2predDict))
            self.__find_rest_pair()
            self.__find_rest_pair()
            if len(self.__std2predDict) > 3:
                dstPoints = np.float32(self.__stdInf[:, :2][list(self.__std2predDict.keys())])
                srcPoints = np.float32(self.__predInf[:, :2][list(self.__std2predDict.values())])
                # ransac
                # transformation_matrix, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, self.__ransacThreshold)

                # estimateAffinePartial2D
                transformation_matrix, mask = cv2.estimateAffinePartial2D(srcPoints[:, :2], dstPoints[:, :2])
                transformation_matrix = np.insert(transformation_matrix, 2, values=[0, 0, 1], axis=0)

                self.__match_point_by_transMat_and_mask(transformation_matrix, mask)
                answer2 = copy.deepcopy(self.__std2predDict)

                self.__std2predDict = answer1 if len(answer1) > len(answer2) else answer2


# mp = matchPoint(stdInf=standardPoints, debug=True)
# ret = mp.match_point(predictPoints)
def parseStdPredInf2List(stdInf):
    sides=set()
    for point in stdInf:
        if point.get('side', 'NO_THIS_KEY') != 'NO_THIS_KEY':
            sides.add(str(point['side']))
    sides = list(sides)
    sides.sort()

    if len(sides)>1:
        stdInfList=[]
        for side in sides:
            stdInfItem=list(filter(lambda x: str(x.get('side', 'NO_THIS_KEY')) == side, stdInf))
            stdInfList.append(stdInfItem)
        return stdInfList
    return [stdInf]


class matchPoint:
    def __init__(self, stdInf, debug=False):
        # self.__stdInf = stdInf
        # self.__predInf = []
        self.__stdInfList = parseStdPredInf2List(stdInf)
        self.__predInfList = []

        self.__debug = debug

    def match_point(self, predInf):
        self.__predInfList = parseStdPredInf2List(predInf)
        assert len(self.__stdInfList)==len(self.__predInfList)
        ans = []
        for i in range(len(self.__stdInfList)):
            sm = subClass_matchPoint(self.__stdInfList[i], self.__debug)
            ans.extend(sm.match_point(self.__predInfList[i]))
        return ans



