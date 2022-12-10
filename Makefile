update:
	git add .
	git commit -m 'update'
	git push
	sh sync.sh || true