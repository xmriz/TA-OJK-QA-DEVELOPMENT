## **Notes**

Yang telah diselesaikan:
1. Scraping Dokumen Regulasi OJK
2. Filtering Dokumen Regulasi OJK, hanyak mengambil dokumen utama (bukan lampiran, faq, ringkasan, dll)
3. Melakukan generate dataset context-question-answer dari dokumen yang telah diambil
4. Melakukan preprocessing dataset context-question-answer, dnenga menghapus data yang memiliki similarity diatas 0.8 pada contextnya.
5. Melakukan pembagian dataset menjadi data SFT, data PPO, dan data uji, masing-masing 60%, 30%, dan 10%
6. Melakukan base evaluation menggunakan data uji yang telah disiapkan, dengan menggunakan berbagai model LLM yang mendukung bahasa Indonesia.

Yang akan dilakukan:
1. Melakukan supervised fine-tuning model SeaLLM 8B dengan menggunakan data SFT
2. Melakukan rewarding data PPO dengan menggunakan gpt-4o-mini
3. Melakukan RLHF PPO training dengan menggunakan data PPO yang telah dihasilkan dari langkah sebelumnya
4. Melakukan evaluation model yang telah dihasilkan dari langkah sebelumnya dengan menggunakan data uji yang telah disiapkan
5. Melakukan deployment model yang telah dihasilkan dari langkah sebelumnya ke dalam bentuk streamlit app