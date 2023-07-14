label_dict = {}

with open('labelmap.txt') as f:
    i = 0
    for row in f:
        label_dict[i] = row
        i+=1

print (label_dict[46]) 