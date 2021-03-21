import pickle

example = {1: "1", 2: '2', 3: '3'}

pickle_out = open('name.pickle', 'wb')
pickle.dump(example, pickle_out)
pickle_out.close()

pickle_in = open('name.pickle', 'rb')
example = pickle.load(pickle_in)

print(example)
print(example[2])

pickle_in.close()