import logging


class Logger(object):
    def __init__(self, log_file_name: str, logger_name: str, log_level=logging.DEBUG):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)
        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)
        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)
        # 创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()
        # 定义handler的输出格式
        # formatter = logging.Formatter(
        #     '[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        # formatter = logging.Formatter('%(asctime)s %(filename)s\t[line:%(lineno)d] %(levelname)s %(message)s')
        # logging.basicConfig(level=level,
        #                     format='[%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p]')
        formatter = logging.Formatter('%(asctime)-10s: %(filename)s, line:%(lineno)d %(levelname)s: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

    def log_params(self, configs: dict):
        msg = "\n Parameters List:"
        for key, value in configs.items():
            msg += f'{key}:{value}\n'
        msg += 'Parameters List End!\n'
        self.__logger.info(msg)


if __name__ == '__main__':
    logger = Logger(log_file_name='./log.txt', log_level=logging.DEBUG, logger_name="point_match_logger").get_log()
    logger.debug('testing ... ')
    logger.info('testing ... ')
