from langchain_experimental.data_anonymizer import PresidioAnonymizer

anonymizer = PresidioAnonymizer()

texto_original = "Mi nombre es Carlos y mi correo es carlos@empresa.com. Llámame al 555-1234."

texto_seguro = anonymizer.anonymize(texto_original)

print(texto_seguro)