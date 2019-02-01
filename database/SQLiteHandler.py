# -*- coding:utf-8 -*-

import sqlite3
import logging
import time
import datetime

class SQLiteHandler(logging.Handler):
	initial_sql = """
	CREATE TABLE IF NOT EXISTS log
			(
			Created text,
			LogLevelName text,    
			Message text
			) 

	"""
	insertion_sql = """
	INSERT INTO log(
					Created,
					LogLevelName,
					Message
                   )
					VALUES (
					'%(dbtime)s',
					'%(levelname)s',
					'%(msg)s'
                   );

	"""


	def __init__(self, db=None):
		logging.Handler.__init__(self)
		self.db = db or "database/huodian.log"
		conn = sqlite3.connect(self.db)
		conn.execute(SQLiteHandler.initial_sql)
		conn.commit()

	def formatDBTime(self, record):
		record.dbtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))

	def emit(self, record):
		self.format(record)
		self.formatDBTime(record)

		sql = SQLiteHandler.insertion_sql % record.__dict__
		conn = sqlite3.connect(self.db)
		conn.execute(sql)
		conn.commit()


#logging.basicConfig(level = logging.INFO,format = qformat, class=SQLiteHandler.SQLiteHandler)

#test = SQLiteHandler()