class MemoryStorage:
    data = {}

    @staticmethod
    def save(storage, key, value):
        if storage not in MemoryStorage.data:
            MemoryStorage.data[storage] = {}

        MemoryStorage.data[storage][key] = value

    @staticmethod
    def load(storage, key):
        return MemoryStorage.data.get(storage, {}).get(key)

    @staticmethod
    def all(storage):
        return MemoryStorage.data.get(storage, {}).items()
    
    @staticmethod
    def exists(storage, key):
        return key in MemoryStorage.data.get(storage, {})