async function getData() {
	const response = await fetch('data.json');
	const json = await response.json();
	return json;
}

let data;

(async () => {
	data = await getData();
	handleChange(10);
})();

function makeRuby(ch, pinyin) {
	const ruby = document.createElement('ruby');
	ruby.appendChild(document.createTextNode(ch));

	const rp_left = document.createElement('rp');
	rp_left.appendChild(document.createTextNode('('));
	ruby.appendChild(rp_left);

	const rt = document.createElement('rt');
	rt.lang = 'zh-Latn';
	rt.appendChild(document.createTextNode(pinyin));
	ruby.appendChild(rt);

	const rp_right = document.createElement('rp');
	rp_right.appendChild(document.createTextNode(')'));
	ruby.appendChild(rp_right);

	return ruby;
}

function toChinese(n) {
	const ch = data[n - 1][0], pinyin = data[n - 1][1];
	return makeRuby(ch, pinyin);
}

function makeChineseMultiplyStringNode(a, b, radix) {
	if (a <= 0 || b <= 0 || a >= radix || b >= radix || a > b) {
		throw new Error('a and b should be in a correct range.');
	}
	const res = a * b;
	if (res < radix) {
		const span = document.createElement('span');
		span.appendChild(toChinese(a));
		span.appendChild(toChinese(b));
		span.appendChild(makeRuby('得', 'dé'));
		span.appendChild(toChinese(res));
		return span;
	} else if (res == radix) {
		const span = document.createElement('span');
		span.appendChild(toChinese(a));
		span.appendChild(toChinese(b));
		span.appendChild(toChinese(1));
		span.appendChild(makeRuby('十', 'shí'));
		return span;
	} else if (res < 2 * radix) {
		const span = document.createElement('span');
		span.appendChild(toChinese(a));
		span.appendChild(toChinese(b));
		span.appendChild(makeRuby('十', 'shí'));
		span.appendChild(toChinese(res - radix));
		return span;
	} else {
		const msb = Math.floor(res / radix), lsb = res % radix;
		if (lsb == 0) {
			const span = document.createElement('span');
			span.appendChild(toChinese(a));
			span.appendChild(toChinese(b));
			span.appendChild(toChinese(msb));
			span.appendChild(makeRuby('十', 'shí'));
			return span;
		} else {
			const span = document.createElement('span');
			span.appendChild(toChinese(a));
			span.appendChild(toChinese(b));
			span.appendChild(toChinese(msb));
			span.appendChild(makeRuby('十', 'shí'));
			span.appendChild(toChinese(lsb));
			return span;
		}
	}
}

function handleChange(radix) {
	const table = document.createElement('table');
	const thead = document.createElement('thead');
	const tr = document.createElement('tr');
	tr.appendChild(document.createElement('th'));
	for (let i = 1; i < radix; i++) {
		const th = document.createElement('th');
		th.appendChild(toChinese(i));
		tr.appendChild(th);
	}
	thead.appendChild(tr);
	table.appendChild(thead);
	const tbody = document.createElement('tbody');
	for (let i = 1; i < radix; i++) {
		const tr = document.createElement('tr');
		const th = document.createElement('th');
		th.appendChild(toChinese(i));
		tr.appendChild(th);
		let j;
		for (j = 1; j <= i; j++) {
			const td = document.createElement('td');
			td.appendChild(makeChineseMultiplyStringNode(j, i, radix));
			tr.appendChild(td);
		}
		for (; j < radix; j++) {
			tr.appendChild(document.createElement('td'));
		}
		tbody.appendChild(tr);
	}
	table.appendChild(tbody);
	document.body.replaceChild(table, document.querySelector('table'));
}
