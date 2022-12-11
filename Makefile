sync:
	sh sync.sh || true

train: clean
	python3 main.py --train

test: clean
	python3 main.py --test

update:
	git add .
	git commit -m 'update'
	git push remote main
	sh sync.sh || true

.PHONY: clean

clean:
	rm logs -rf
	rm output -rf