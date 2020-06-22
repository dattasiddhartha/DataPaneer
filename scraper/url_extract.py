from urlextract import URLExtract
import pickle
import re

extractor = URLExtract()
urls = []
matchstr = 'https://www.allrecipes.com/recipe/'

lines = []
f = open('urls.txt')

while(True):
	try:
		line = f.readline()
		if(not line):
			break
		lines.append(line)
	except:
		pass

print('num lines', len(lines))

for line in lines:
	[urls.append(url) for url in re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', line) if (url[:len(matchstr)] == matchstr)]
	#[urls.append(url) for url in extractor.find_urls(line) if (url[:len(matchstr)] == matchstr)]

urls = list(set(urls))
print("Saved", len(urls), "urls")
pickle.dump( urls, open( "sites2.p", "wb" ))