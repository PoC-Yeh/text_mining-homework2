import pandas as pd


#test data
with open("/Users/AllisonYeh/TMU/text_mining/PA-3/TestData.txt", 'r', encoding='UTF-8') as c:
    test_data = c.readlines()

#train data    
with open("/Users/AllisonYeh/TMU/text_mining/PA-3/TrainingData.txt", 'r', encoding='UTF-8') as c:
    train_data = c.readlines()
    
#punctuations
with open("/Users/AllisonYeh/TMU/text_mining/PA-2/punctuation.txt", 'r', encoding='UTF-8') as c:
    pun = list(c.read())
punctuation = [p for p in pun if pun.index(p) %2 == 0]
#print(punctuation)

#stopwords
with open("/Users/AllisonYeh/TMU/text_mining/PA-2/stopword_chinese.txt", 'r', encoding='UTF-8') as c:
    stop = c.readlines()
stop_word = [s.strip() for s in stop]
#print(stop_word)


#######################deal with train data
sports_total_word_count = 0
politics_total_word_count = 0
doc_term = {}
term_count_sports_politics = {"sports_word":{}, "politics_word": {}}
sports_doc_count = 0
politics_doc_count = 0

for count in range(1, 3501):
    test = train_data[count - 1].replace("\u3000", "").split(" ")[:-1]
    test[0] = test[0].split("\t")[1]
    new_test = [t for t in test if len(t) > 0]
    f1 = list(filter(lambda x: x not in punctuation, new_test))
    f2 = list(filter(lambda x: x not in stop_word, f1))
    
    doc_term[count] = {}
    for i in f2[1:]:
        if i in list(doc_term[count].keys()):
            doc_term[count][i] += 1
        else:
            doc_term[count][i] = 1
    if f2[0] == "sports":
        #label count
        sports_doc_count += 1
        
        #calculate total doc terms
        for value in list(doc_term[count].values()):
            sports_total_word_count += value
            
        #calculate total count of each word in sports documents
        for x in list(doc_term[count].keys()):
            if x in list(term_count_sports_politics["sports_word"].keys()):
                term_count_sports_politics["sports_word"][x] += doc_term[count][x]
            else:
                term_count_sports_politics["sports_word"][x] = doc_term[count][x]
            
    elif f2[0] == "politics":
        #label count
        politics_doc_count += 1
        
        #calculate total doc terms
        for value in list(doc_term[count].values()):
            politics_total_word_count += value
        
        #calculate total count of each word in politics documents
        for x in list(doc_term[count].keys()):
            if x in list(term_count_sports_politics["politics_word"].keys()):
                term_count_sports_politics["politics_word"][x] += doc_term[count][x]
            else:
                term_count_sports_politics["politics_word"][x] = doc_term[count][x]
              
              
print(sports_total_word_count)
print(politics_total_word_count)
print(len(list(doc_term.keys()))) #3500
print(len(list(term_count_sports_politics["sports_word"].keys())))
print(len(list(term_count_sports_politics["politics_word"].keys())))
print(sports_doc_count)
print(politics_doc_count) 

doc_term_df = pd.DataFrame(doc_term)
SP_term_count = pd.DataFrame(term_count_sports_politics)
new_df = doc_term_df.join(SP_term_count)
new_df.fillna(0, inplace = True)

vocabulary = len(new_df.index)
new_df["sports_P"] = (new_df["sports_word"] + 1) / (sports_total_word_count + vocabulary)
new_df["policts_P"] = (new_df["politics_word"] + 1) / (politics_total_word_count + vocabulary)
                

#######################deal with test data
doc_term_2 = {}

for count in range(1, len(test_data)):
    test = test_data[count - 1].replace("\u3000", "").split(" ")[:-1]
    doc_id = test[0].split("\t")[0]
    test[0] = test[0].split("\t")[1]
    new_test = [t for t in test if len(t) > 0]
    f1 = list(filter(lambda x: x not in punctuation, new_test))
    f2 = list(filter(lambda x: x not in stop_word, f1))

    doc_term_2[doc_id] = {}
        
    for i in f2:
        if i in list(doc_term_2[doc_id].keys()):
            doc_term_2[doc_id][i] += 1
        else:
            doc_term_2[doc_id][i] = 1
        
        
test_df = pd.DataFrame(doc_term_2)
test_df.fillna(0, inplace = True)
sport_p = new_df["sports_P"]
politics_p = new_df["policts_P"]

test_df_v2 = test_df.join(sport_p)
test_df_v3 = test_df_v2.join(politics_p)
test_df_v3.fillna(0, inplace = True)
test_df_v3


#calculate P(sports) and P(politics) of each test doc
sports_ratio = sports_doc_count / 3500
politics_ratio = politics_doc_count /3500

test_doc_classify = []
for doc_id in list(doc_term_2.keys()):
    sports_probability = 1 * sports_ratio
    politics_probability = 1 * politics_ratio
    for token in test_df_v3.index:
        if test_df_v3[doc_id][token] > 0 and test_df_v3["sports_P"][token] > 0:
            log_sports_p = math.log10(test_df_v3["sports_P"][token])
            sports_probability *= test_df_v3[doc_id][token] * log_sports_p
        
        if test_df_v3[doc_id][token] > 0 and test_df_v3["policts_P"][token] > 0:
            log_politics_p = math.log10(test_df_v3["policts_P"][token])
            politics_probability *= test_df_v3[doc_id][token] * log_politics_p
    
    inside = [int(doc_id)]
    if sports_probability > politics_probability:
        inside.append("sports")
    elif sports_probability < politics_probability:
        inside.append("politics")
    elif sports_probability == politics_probability:
        inside.append("P(sports) = P(politics)")
    
    test_doc_classify.append(inside)

sorted_list = sorted(test_doc_classify)


#result output
result_list = ["ID Similarity\n"]
for i in sorted_list:
    inside_list = []
    inside_list.append(str(i[0]))
    inside_list.append(str(i[1]))
    new = " ".join(inside_list)+ "\n"
    result_list.append(new)
    
print(result_list[:11])

f = open('classify_result.txt', 'w', encoding = 'UTF-8') 
for i in result_list:
    f.write(i)
f.close()
