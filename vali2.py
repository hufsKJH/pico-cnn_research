text_file_path = './NEWvalidation.txt'
new_text_file_path = './label.txt'
new_text_content= ''

with open(text_file_path,'r') as f:
	lines = f.readlines()
	for i, l in enumerate(lines):
		new_text = l.split(' ')
		new_string = new_text[-1]
		new_text_content += new_string
		
with open(new_text_file_path,'w') as f:
	f.write(new_text_content)

