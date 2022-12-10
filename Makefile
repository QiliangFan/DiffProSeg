update:
	git add .
	git commit -m 'update'
	git push remote main
	sh sync.sh || true