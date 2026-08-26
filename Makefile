CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c99
PREFIX  ?= $(shell brew --prefix 2>/dev/null)
BUILD   := build

# Homebrew keeps headers and libraries outside the default search path on macOS.
ifneq ($(PREFIX),)
CFLAGS  += -I$(PREFIX)/include
LDFLAGS += -L$(PREFIX)/lib
endif

all: $(BUILD)/bmagent

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/bmagent: bmagent.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lcurl -lcjson

clean:
	rm -rf $(BUILD)

.PHONY: all clean
