text_file_path = './validation.txt'
new_text_file_path = './NEWvalidation.txt'
new_text_content = ''

target1 = 'mv '
target = '/'
new_word = ''

with open(text_file_path,'r') as f:
	lines = f.readlines()
	for i, l in enumerate(lines):
		new_string=l.strip().replace(target,new_word)
		new_string=new_string.strip().replace(target1,new_word)
		if new_string:
			new_text_content += new_string+'\n'
		else:
			new_text_content += '\n'

with open(new_text_file_path,'w') as f:
	f.write(new_text_content)
