import pickle

handle_w = open('filename.pickle', 'wb')

a = {'hello': 'world'}
pickle.dump(a, handle_w, protocol=pickle.HIGHEST_PROTOCOL)

b = {'hello': 'you'}
pickle.dump(b, handle_w, protocol=pickle.HIGHEST_PROTOCOL)

handle_w.close()

handle_r = open('filename.pickle', 'rb')
print(pickle.load(handle_r)['hello'])

handle_r.close()