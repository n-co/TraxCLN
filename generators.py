import numpy as np
import time
import datetime as dt

start_time = dt.datetime.now().replace(microsecond=0)

def primes(limit=None):
    i = 2
    ps = []
    while limit is None or i < limit:
        is_prime = True
        for p in ps:
            if i % p == 0:
                is_prime = False
                break
            if p > np.sqrt(i):
                break
        if is_prime:
            yield i
            ps.append(i)
        i += 1


def seven_boom(limit=None):

    def contains7(num):
        while num > 1:
            if num % 10 == 7:
                return True
            num //= 10
        return False

    i = 1
    while limit is None or i < limit:
        if i % 7 == 0 or contains7(i):
            yield 'boom'
        else:
            yield i
        i += 1

for p in primes(300000):
    p = 5

end_time = dt.datetime.now().replace(microsecond=0)

print start_time.strftime("%d-%m-%Y_%H-%M-%S")
print 'start time:    %s' % start_time
print 'end time:      %s' % end_time
print 'total runtime: %s' % (end_time - start_time)

l = [i**2 for i in xrange(0, 10)]
g = (i**2 for i in xrange(0, 10))
