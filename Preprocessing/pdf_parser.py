# documentation for SciPDF and GROBID at: https://github.com/titipata/scipdf_parser
import scipdf
import os


directory = 'pdf/devset'
print("Start parsing...")
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        article_dict = scipdf.parse_pdf_to_dict(os.path.join(dirpath, filename), soup=True)

        methods_section = []
        start_adding = False

        for section in article_dict['sections']:
            heading = section['heading'].lower()

            if "methods" in heading.lower() or "method" in heading.lower():
                start_adding = True

            if "results" in heading.lower():
                break
            
            if start_adding:
                methods_section.append(section)

        with open("txt/" + filename.split(".")[0] + ".txt", 'w') as f:
            for section in methods_section:
                f.write(section['heading'])
                f.write("\n")
                f.write(section['text'])
                f.write("\n")
                f.write("\n")
        print("finished with exporting " + filename)