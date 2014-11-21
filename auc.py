
import sys
''''
input :
	0,0.03612024
	0,0.04124144
	0,0.03739638
	1,0.05071424
	0,0.04080498
	0,0.03288555
	0,0.03816423
	0,0.03296028
	0,0.03791010
	0,0.04096453

usage :
	 cat test_imgclass_out.tmp |python auc.py 1
'''


def roc():
	if len(sys.argv) == 2:
		cp = int(sys.argv[1])
	else:
		cp = 1
	data = []
	count = [0,0]
	for line in sys.stdin:
		lable,score = line.strip().split(',')
		lable = int(lable)
		score = float(score)
		count[lable] += 1
		data.append([score,lable])
		
	data = sorted(data,key=lambda x:x[0],reverse=True)

	n = len(data)
	k = 0
	for i in xrange(n):
		if data[i][1] == cp:
			k += n-i
	print (k-count[cp]*(count[cp]+1)/2.0)/(count[0]*count[1])

roc()