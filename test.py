import logging

FORMAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
LEVEL = logging.DEBUG
FILENAME = 'test.log'
FILEMODE = 'a'

logger = logging
logger.basicConfig(
    level = LEVEL,
    format = FORMAT,
    filename = FILENAME,
    filemode = FILEMODE
)


logger.info('info 信息')