import datetime as dt

now = dt.datetime.now()
f = open("tmp.txt", 'a')
f.write(now.strftime("%d.%m.%Y") + "\n")

print "bye"

