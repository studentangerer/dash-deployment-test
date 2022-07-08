class TestClass:
    some_var = 0

def modify():
    TestClass.some_var = 1
    print(TestClass.some_var)

if __name__ == '__main__':
    print(f"Before: {TestClass.some_var}")
    modify()
    print(f"After: {TestClass.some_var}")

