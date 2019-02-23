# -*- coding: utf-8 -*-

import logging
import logging.config
logging.config.fileConfig("log_config.config")

logger = logging.getLogger("test")
logger.info("导入了文件，路径为：")
logger.info("info, ~~!`~!@~!2!~@~!#$#@$#@")
logger.warn("warn,!@#$%#$%#$%#$#$")
