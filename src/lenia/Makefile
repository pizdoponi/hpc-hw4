CC = mpicc
CFLAGS = -O3 -Wall -Wextra -std=c11
LDLIBS = -lm

SRC_DIR = src
TARGET = lenia.out
SRCS = $(wildcard $(SRC_DIR)/*.c)
HDRS = $(wildcard $(SRC_DIR)/*.h)
OBJS = $(SRCS:$(SRC_DIR)/%.c=%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDLIBS)

%.o: $(SRC_DIR)/%.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
