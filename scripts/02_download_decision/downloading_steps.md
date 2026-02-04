# Implementacia stahovania sudnych rozhodnuti z krajskych sudov, najvyssieho a ustavneho sudu

## Co sme  riesili
Nasim cielom bolo automaticky stiahnut texty sudnych rozhodnuti a PDF prilohy na zaklade zoznamu spisovych znaciek (napr. "14Cob/20/2019"). Presli sme si niekolkymi zdrojmi a narazili na rozne typy ochran, kym sme nasli funkcne riesenie.

## 1. CRZ
Vacsina ludi vyuziva centralny register zmluv, co je naozaj skvely pristup, avsak my hladame sudne rozhodnutia a nie zmluvy. Z toho dovodu sme sa tym ani nezaoberali.


## 2. Pokus: Slov-Lex (neuspesne)
Najprv sme skusali tahat data z portalu `slov-lex.sk`. Tu sme narazili na problem s technologiou **Liferay**.

* **Co je to Liferay:** Je to masivny portalovy system (nieco ako enterprise verzia WordPressu), ktory pouziva stat. Problem je, ze Liferay si velmi strazi "relacie" (sessions).
* **Preco mi to neslo:** System vyzaduje dynamicky generovany kluc (`Liferay.authToken`), ktory sa meni kazdych par minut. Ked sme poslali poziadavku cez Python, server zistil, ze nemame tento token ani cookies realneho prehliadaca. Namiesto JSON dat nam vratil "soft block" – cize obycajne HTML uvodnej stranky, tvaril sa, ze nic nenasiel. Obchadzat to by znamenalo simulovat celeho uzivatela, co je nestabilne.


## 3. Pokus: Otvorene Sudy (neuspesne)
Dalej sme skusili web `otvorenesudy.sk`, ktory agreguje data. Tu nas zablokoval **Cloudflare**.

* **Cloudflare:** ako vsetci asi pozname je sluzba, ktora funguje ako "vrator" pred webovou strankou. Chrani server pred utokmi a robotmi.
* **Preco nam to neslo:** Nas skript okamzite dostal chybu **HTTP 403 Forbidden**. Cloudflare detegoval, ze poziadavka nejde z bezneho prehliadaca (ako Chrome), ale z programovacieho jazyka. Vyhodil nam stranku s Captchou ("Verify you are human"), ktoru automaticky skript nevie vyplnit bez zloziteho pouzitia nastrojov ako Playwright v vizualnom rezime.


## 4. Finalne riesenie: API Ministerstva spravodlivosti (ispesne)
Nakoniec sme nasli cestu cez `obcan.justice.sk`. Konkretne sme vyuzili ich **API**, ktore nie je chranene tak agresivne ako frontendove weby.

* **Preco to funguje:** Toto API sluzi pre interne systemy ministerstva. Vracia nam data vo formate **JSON** (strukturovana textova odpoved), kde mame presne vsetko, co potrebujeme – metadata aj priame linky na stiahnutie dokumentov.

### Ako funguje nas finalny skript:
1.  **Vstup:** Nacitame CSV subor so znackami.
2.  **Generovanie variantov:** Kedze sudcovia pisu znacky rozne (raz s lomkou, raz s medzerou), skript si vygeneruje vsetky mozne kombinacie IDcka (napr. `14Cob 20 2019` aj `14Cob/20/2019`).
3.  **Hladanie v API:** Posielame poziadavky na endpoint `/v1/rozhodnutie`. Ak najdeme zhodu v spise aj nazve sudu, berieme to.
4.  **Stahovanie:** Z JSON odpovede vytiahneme ID dokumentu a stiahneme ho ako PDF alebo DOCX.


## 5. Pokus: Oficialny web Ustavneho sudu (neuspesne)
Toto bolo technikcky najnarocnejsie. Rozhodnutia Ustavneho sudu nie su v beznom API Ministerstva spravodlivosti, takze sme museli ist priamo na ich web (`ustavnysud.sk`) alebo na agregator (`otvorenesudy.sk`). Vyskusal sme vsetky dostupne metody o ktorych viem a ktore som si dohladal.
### Pokus A: Klasicky Python request (Library `requests`)
* **Co sme spravili:** Poslali sme na ich vyhladavanie beznu HTTP poziadavku, aka sa pouziva pri 90% scrapingov.
* **Vysledok:** Okamzita chyba **403 Forbidden**.
* **Dovod:** Server okamzite spoznal, ze sa pytame cez Python skript (podla `User-Agent` hlavicky) a spojenie zarezla.

### Pokus B: Pokrocila impersonacia (Library `curl_cffi`)
* **Co sme spravili:** Kedze obycajny request nepresiel, nasadili sme specialnu kniznicu `curl_cffi`. Ta dokaze simulovat nielen hlavicky, ale aj takzvanu "TLS Fingerprint" (odtlacok prehliadaca). Nastavili sme ju, aby sa tvarila presne ako Chrome 110.
* **Vysledok:** Znova chyba **403 Forbidden**.
* **Dovod:** Toto bolo prekvapenie. Znamena to, ze server US SR ma nasadene velmi agresivne ochrany (WAF - Web Application Firewall), ktore pravdepodobne blokuju pristup uz na urovni siete alebo deteguju, ze request ide z podozrivej IP adresy (napr. skolskej siete alebo VPN), pripadne analyzuju spravanie na TCP urovni.

### Pokus C: Agregator Otvorene Sudy (Library `Playwright`)
* **Co sme spravili:** Kedze oficialny web nesiel, skusili sme to obklukou cez `otvorenesudy.sk`. Pouzili sme nastroj **Playwright**, ktory spusti skutocny prehliadac (Chromium) a ovlada ho kodom.
* **Vysledok (Headless rezim):** Ked bezal prehliadac skryto na pozadi, **Cloudflare** nas okamzite odhalil a nepustil k obsahu.
* **Vysledok (Headless=False):** Ked sme prehliadac otvorili viditelne, Cloudflare nam vyhodil **Captchu** ("Verify you are human").
* **Preco to neslo automatizovat:** Museli by sme rucne klikat na Captchu pri kazdom spusteni alebo platit drahe sluzby na jej prelomenie. To pre nas ucely (masove stahovanie) nie je efektivne.



## Vysledok
Mame funkcny downloader pre Najvyssi sud a vsetky Krajske sudy. Ustavny sud zatial riesit nebudeme, kedze nie je v tejto databaze a jeho vlastny web je prilis silno chraneny, takisto sudnych rozhodnuti ktore sa zaoberaju moderacnym pravom na ustavnom sude nie je tolko vela takze zatial to nechame tak. 
Nikdy nebude potrebne stahovat vela rozhonduti z US.