# pdf documents manipulation
from fpdf import FPDF

# Create a .txt file and add models metrics table to it
def write_results_txt(txt_metrics_table:str) -> None:
    with open('reports/results.txt', 'w') as file:
        file.write('MODELS RESULTS FOR 10 FOLDS:\n')
        file.write(txt_metrics_table)

# Create a .pdf file and add models metrics table to it
def write_results_pdf(metrics_table:list) -> None:    
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, txt="MODELS RESULTS FOR 10 FOLDS", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.write_html(
        f"""
        <table border="1">
            <thead>
                <tr>
                    <th width="15%">{metrics_table[0][0]}</th>
                    <th width="15%">{metrics_table[0][1]}</th>
                    <th width="15%">{metrics_table[0][2]}</th>
                </tr>
            </thead>
            
            <tbody>
                <tr>
                    <td>{'</td><td>'.join([str(value) for value in metrics_table[1]])}</td>
                </tr>

                <tr>
                    <td>{'</td><td>'.join([str(value) for value in metrics_table[2]])}</td>
                </tr>
            </tbody>
        </table>""",
        table_line_separators=True,
    )

    pdf.output("reports/results.pdf")