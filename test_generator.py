def generator():
    while True:
        yield 0


for idx, val in enumerate(generator()):
    print(idx, val)
