mecab_dict_index = @MECAB_DICT_INDEX@
dicdir           = @MECAB_DICDIR@

# SUBDIRS    = doc script
dic_DATA   = @MECAB_GENDATA@ @MECAB_LEXICAL_DIC@ @MECAB_PREDATA@
EXTRA_DIST = @MECAB_LEXICAL_DIC@ @MECAB_PREDATA@ RESULT
CLEANFILES = @MECAB_GENDATA@

@MECAB_GENDATA@:
	$(mecab_dict_index) -d . -o . -f EUC-JP -t @CHARSET@
	@echo To enable dictionary, rewrite @MECAB_MECABRC@ as \"dicdir = @MECAB_DICDIR@\"

rpm: dist
	rpm -ta @PACKAGE@-@VERSION@.tar.gz

export-package:
	./upload.pl -p mecab -n @PACKAGE@ -r @VERSION@ -f @PACKAGE@-@VERSION@.tar.gz

install-exec-hook:
	if ! [ -d $(DESTDIR)/etc/mecab/dic/naist-jdic ]; \
		then mkdir -p $(DESTDIR)/etc/mecab/dic/naist-jdic; \
	fi
	if ! [ -f $(DESTDIR)/etc/mecab/dic/naist-jdic/dicrc ]; \
		then $(LN_S) @MECAB_DICDIR@/dicrc $(DESTDIR)/etc/mecab/dic/naist-jdic/dicrc; \
	fi
