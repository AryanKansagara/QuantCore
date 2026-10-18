CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wpedantic

.PHONY: all test sanitize benchmark clean
all: build/quantcore build/tests

build:
	mkdir -p build

build/quantcore: benchmark.cpp quant_core.hpp | build
	$(CXX) $(CXXFLAGS) benchmark.cpp -o $@

build/tests: tests.cpp quant_core.hpp | build
	$(CXX) $(CXXFLAGS) tests.cpp -o $@

build/tests-sanitized: tests.cpp quant_core.hpp | build
	$(CXX) -std=c++17 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer tests.cpp -o $@

test: build/tests
	./build/tests

sanitize: build/tests-sanitized
	./build/tests-sanitized

benchmark: build/quantcore
	./build/quantcore

clean:
	$(RM) build/quantcore build/tests build/tests-sanitized
