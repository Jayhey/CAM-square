import csv
import pandas as pd
import json
import random
import time
import datetime
import os
import collections
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import accuracy_score
from CNN_module import *

max_input_length = 30
embedding_dim = 100
filter_sizes = [3,4,5]
num_filters = 128
dropout_keep_prob = 0.5
num_classes = 20

evaluate_every = 10
num_epochs = 15
batch_size = 32
lr = 1e-3

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "graph", timestamp))
result_dir = os.path.abspath(os.path.join(os.path.curdir, "result"))

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

with open('original_data.json', encoding='utf-8') as f:
    data = json.load(f)

np.seterr(divide='ignore', invalid='ignore')

random.seed(1234)
random.shuffle(data)

df = pd.DataFrame(data)
cat = pd.factorize(df.cuisine)
df.cuisine = pd.factorize(df.cuisine)[0]

idx = int(len(data) * 0.8)
train = pd.DataFrame(df[:idx])
test = pd.DataFrame(df[idx:])

ingredient_list = list(df['ingredients'])
unique_ingredient = ['<UNK>'] + list(set(x for l in ingredient_list for x in l))
unique_df = pd.DataFrame(np.array(unique_ingredient))

x_train = np.stack(map_func(train['ingredients'], unique_ingredient, max_input_length), 0)
x_test = np.stack(map_func(test['ingredients'], unique_ingredient, max_input_length), 0)

y_train = np.eye(num_classes)[train.cuisine.values]
y_test = np.eye(num_classes)[test.cuisine.values]

print("Preprocessing complete!")

sess = tf.Session()
model = TextCNN(sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(unique_ingredient),
                embedding_size=embedding_dim,
                filter_sizes=filter_sizes,
                num_filters=num_filters)

# Define Training procedure
optimizer = model.adam

train_writer = tf.summary.FileWriter(os.path.join(out_dir, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(out_dir, 'test'))

print("Writing to {}\n".format(out_dir))
sess.run(tf.global_variables_initializer())


batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs, shuffle=True)

testpoint = 0

start_time = time.time()
for batch in batches:
    x_batch, y_batch = zip(*batch)    
    summary, _, acc, loss_ = sess.run([model.merged_summary_op, 
                                       optimizer, 
                                       model.accuracy, 
                                       model.loss],
                                      feed_dict={model.input_x: x_batch,
                                                 model.input_y: y_batch,
                                                 model.learning_rate: lr,
                                                 model.dropout_keep_prob: dropout_keep_prob})
    step = tf.train.global_step(sess, model.global_step)
    
    
    if step % evaluate_every == 0:
        time_str = datetime.datetime.now().isoformat()
        print("train: {}, step {}, loss {:g}, acc {:g}".format(time_str, step, loss_, acc))
        train_writer.add_summary(summary, step)
        if testpoint + 100 < len(x_test):
            testpoint += 100
        else:
            testpoint = 0
        summary, acc, loss_ = sess.run([model.merged_summary_op, 
                                        model.accuracy, 
                                        model.loss],
                                       feed_dict={model.input_x: x_batch,
                                                  model.input_y: y_batch,
                                                  model.dropout_keep_prob: 1.})
        #print("\nEvaluation:")
        print("test: {}, step {}, loss {:g}, acc {:g}".format(time_str, step, loss_, acc))
        test_writer.add_summary(summary, step)
        print("")

print("{:.1f} seconds".format(time.time()-start_time))


# Embedding ingredients
unique_df.to_csv(os.path.join(out_dir,"test","ingredients.csv"), header=False, index=False)

config=projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = "ingredients.csv"

projector.visualize_embeddings(test_writer, config)
saver_embed = tf.train.Saver([model.We])
saver_embed.save(sess, os.path.join(out_dir,'embedding.ckpt'), 1)


# Inference (calculate test accuracy)
pred_list = np.array([])
for i in range(len(x_test)//100+1):
    if i == len(x_test)//100:
        x_batch, y_batch = x_test[i*100:], y_test[i*100:]
    else:
        x_batch, y_batch = x_test[i*100:(i+1)*100], y_test[i*100:(i+1)*100]

    step, pred,loss, accuracy = sess.run([model.global_step, 
                                          model.predictions, 
                                          model.loss, 
                                          model.accuracy],
                                         feed_dict = {model.input_x: x_batch,
                                                      model.input_y: y_batch,
                                                      model.dropout_keep_prob: 1.0
                                                     })
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    pred_list = np.append(pred_list, pred)
test_acc = accuracy_score(test.cuisine.values, pred_list)
print("Test accuracy : {}".format(test_acc))

# Extract CAM score

fin_weights = sess.run(tf.trainable_variables()[-2])
batches = batch_iter(list(zip(x_test, y_test)), batch_size, 1, shuffle=False)
results = []
doc_idx = 0

for num, batch in enumerate(batches):
    x_batch, y_batch = zip(*batch)
    if len(x_batch) == batch_size:
        actmaps, predictions = sess.run([model.h_outputs, model.predictions],
                                                feed_dict={model.input_x: x_batch,
                                                           model.input_y: y_batch,
                                                           model.dropout_keep_prob: 1.0})

        for batch_idx in range(batch_size):
            fin_result = collections.OrderedDict()
            cook = test.ingredients.values[doc_idx]
            for idx in range(len(actmaps)):
                combined_actmap = actmaps[idx][batch_idx].reshape(
                    (max_input_length+idx, num_filters))    

                batch_result = np.dot(combined_actmap, fin_weights[idx*128:(idx+1)*128,:])
                batch_result = batch_result[:, predictions[batch_idx]]

                for ing_idx, word in enumerate(cook):                      
                    score = np.mean(batch_result[ing_idx:ing_idx+filter_sizes[idx]])
                    fin_result[cook[ing_idx]] = score

            preinfo = {'실제값' : cat[1][np.argmax(y_test[doc_idx])], 
                       '예측값' : cat[1][predictions[batch_idx]]}
            results.append([preinfo, fin_result])
            doc_idx += 1

with open(os.path.join(result_dir,'result.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for doc in results:
        writer.writerow([doc])



# 상위 5개에 별표 치기 - star_results

star_results = []
for result_idx, data in enumerate(results[3:]):
    tmp = sorted(data[1].items(), key=lambda x: x[1], reverse=True)[0:5] ##
    tmp = [score_tuple[0] for score_tuple in tmp]
    star_result = collections.OrderedDict()
    for score_tuple in data[1].items():
        star_result[score_tuple[0]] = ''
    for num in range(len(tmp)):
        star_result[tmp[num]] = '*' * (num + 1)
    star_results.append([data[0],star_result])

with open(os.path.join(result_dir, 'star_results.csv'), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    for doc in star_results:
        writer.writerow([doc])

# 상위 5개에 포함돼 있는 단어 사전 만들기
def count(cuisine):
    cuisine_dict = collections.defaultdict(int)
    for result_idx, data in enumerate(results):
        tmp = sorted(data[1].items(), key=lambda x: x[1], reverse=True)[0:5] ##
        tmp = [score_tuple[0] for score_tuple in tmp]
        if data[0]['예측값'] == cuisine:
            for num in range(len(tmp)):
                if type(tmp[num]) == int:
                    continue
                cuisine_dict[tmp[num]] += 1
    return sorted(cuisine_dict.items(), key=lambda x: x[1], reverse=True)

top_dict = {}
for cui in cat[1]:
    top_dict[cui] = count(cui)

# save as json
with open(os.path.join(result_dir,'top_dict.json'), 'w') as fp:
    json.dump(top_dict, fp)