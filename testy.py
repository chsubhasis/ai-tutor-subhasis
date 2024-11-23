import sqlite3
import sys
import pysqlite3

sys.modules['sqlite3'] = pysqlite3

print(sqlite3.sqlite_version)
print(sqlite3.__file__)
#print(pysqlite3)
print(sys.modules['sqlite3'])