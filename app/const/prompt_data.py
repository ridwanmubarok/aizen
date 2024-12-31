PROMPT_LANG_FORMAT = {
    'en': '''
        A patient has been diagnosed with {disease} with a confidence level of {confidence:.1%}. 
        The symptoms identified in this patient are as follows: {symptoms}. 
        The patient is located in {location} and is currently experiencing the {season} season.
        Based on the available data, please prepare a report that can be used as a reference for healthcare professionals. 
        This report should include:
        - Further explanation regarding the disease
        - Recommendations for traditional and modern medications
        - Addresses of clinics, hospitals, other healthcare facilities, or places that can be considered for healthcare professionals
        - First aid suggestions
        - Over-the-counter medications that can be purchased at pharmacies
    ''',
    
    'id': '''
        Seorang pasien didiagnosis menderita {disease} dengan tingkat keyakinan {confidence:.1%}. 
        Gejala yang teridentifikasi pada pasien ini adalah sebagai berikut: {symptoms}. 
        Pasien berada di lokasi: {location} dan saat ini sedang mengalami musim {season}.
        Berdasarkan data yang ada, mohon untuk menyusun laporan yang dapat digunakan sebagai bahan pertimbangan bagi tenaga kesehatan. 
        Laporan ini harus mencakup:
        - Penjelasan lebih lanjut mengenai penyakit
        - Rekomendasi obat tradisional dan modern
        - Alamat klinik, rumah sakit, fasilitas kesehatan lainnya, atau tempat lain yang dapat dijadikan bahan pertimbangan untuk tenaga kesehatan.
        - saran penanganan pertama
        - produk obat yang dapat dijual secara umum dan dapat dibeli di apotik
    ''',
    
    'ja': '''
        患者は、{disease}と診断され、信頼度は{confidence:.1%}です。患者に確認された症状は以下の通りです：{symptoms}。
        患者は{location}におり、現在{season}の季節を迎えています。
        利用可能なデータに基づいて、医療専門家の参考として使用できる報告書を作成してください。この報告書には以下を含める必要があります：
        - 病気に関するさらなる説明
        - 伝統的および現代の薬の推奨
        - 医療専門家が考慮できるクリニック、病院、その他の医療施設の住所
        - 初期対応の提案
        - 薬局で購入できる市販薬
    '''
}
