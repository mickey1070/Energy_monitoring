import mysql.connector

databse=mysql.connector.connect(
    host = 'localhost',
    user='root',
    passwd='password',
    auth_plugin='mysql_native_password'
)

cursorobject=databse.cursor()
cursorobject.execute("CREATE DATABASE energy")
print("all done")