#!coding=utf-8
import os
import sys
from shutil import copyfile


def copy(dirname,name,outpath):
	if not os.path.exists(outpath):
		os.makedirs(outpath)
	inpath = os.path.join(dirname,name)
	outpath = os.path.join(outpath,name)
	copyfile(inpath,outpath)
	
	

def visit(para,dirname,names):
	format = para[0]
	outroot = para[1]
	outpath = os.path.join(outroot,dirname)
	for name in names:
		if format in name:
			print name
			copy(dirname,name,outpath)

def run():
	if len(sys.argv) != 4:
		print "Usage: python copy_files.py [path] [filename] [outpath]"
		return
	path = sys.argv[1]
	para = sys.argv[2:4]	
	os.path.walk(path,visit,para)
	
if __name__ == "__main__":	
	run()