import os
import re
Training_Folder = 'labeled_articles/'
Training_Label = []
Training_Matrix = []
for file in os.listdir(Training_Folder):
        print (file)
        Text_File = open(Training_Folder+file,'r+',encoding='utf-8')
        for line in Text_File:
            if (line.startswith('###')):
                                continue
            #### Sentence preprocessing for cleaning
            #line = line.replace('\n','');line = line.lower()
            line = re.sub(r'[-*+%$()\.,/?!><"&#\[\]\(\)\\]', ' ',line)
            lines = line.split('	')
            #### Putting lines into the list for classification
            Training_Matrix.append(lines[1])
            #### Making file name as label for classification
            Training_Label.append(lines[0]) 
        Text_File.close()
print(Training_Label)
print(Training_Matrix)