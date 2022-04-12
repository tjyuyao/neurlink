from typing_extensions import Self


def test_new_inheritance():

    class Base:

        def __new__(cls: type[Self]) -> Self:
            return None
        
    class Child1(Base):
        ...

    class Child2(Base):

        def __new__(cls: type[Self]) -> Self:
            return object.__new__(cls)
    
    assert Base() is None
    assert Child1() is None
    assert Child2() is not None


def test_class_getitem():

    class Dummy:

        def __new__(cls, *args, **kwds):
            print(args)
            return cls[-1]

        def __class_getitem__(self, key):
            def argparser(*args, **kwds):
                print(key, args, kwds)
            return argparser
    
    print(Dummy(1))

if __name__ == "__main__":
    test_class_getitem()