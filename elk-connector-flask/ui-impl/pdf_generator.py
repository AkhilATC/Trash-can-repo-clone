from datetime import datetime
from xhtml2pdf import pisa
import socket



def __generate_pdf_attributes(clinic_name,
                              clinic_logo,
                              physician_name,
                              physician_address,
                              patient_name,
                              patient_dob,
                              patients_contact,
                              c_complaint,
                              note):
    time_ = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)

    html_ = f"""<header> \
    <h3 align=\"center\" style=\"color:blue;\"> Patient report </h3>\
    <p align="center">System generated report</p> \
    <center> <img src=\"./static/doc-logo.jpeg\" width="50" height="50"> </center>\
    </header>\
    <div style="width: 100%;background-color: #DCDCDC;\
    color:black;padding: 14px 20px;margin: 8px 0;border: none; border-radius: 4px;cursor: pointer; display: table;">\
    <table style="font-family: arial, sans-serif ;border-collapse: collapse; width: 100%;">\
    <tr>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Clinic Name: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {clinic_name} </td>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Clinic logo: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {clinic_logo} </td>\
    </tr>\
    <tr>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Physician Name: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {physician_name} </td>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Physician address: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {physician_address} </td>\
    </tr>\
    <tr>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Patient Name: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {patient_name} </td>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Patient's DOB: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {patient_dob} </td>\
    </tr>\
    <tr>\
    <td style="border-style: none;text-align: left;padding: 8px;"> Patient's Contact: </td>\
    <td style="border: 1px solid #dddddd;text-align: left;padding: 8px;"> {patients_contact} </td>\
    </tr>\
    <tr></tr>
    <tr>\
    <td style="border: 1px solid #799bb0;text-align: left;padding: 8px;"> chef complaint: </td>\
    <td style="border: 1px solid #799bb0;text-align: left;padding: 8px;"> {c_complaint} </td>\
    </tr>\
    <tr>\
    <td style="border: 1px solid #799bb0;text-align: left;padding: 8px;"> Consultation Note: </td>\
    <td style="border: 1px solid #799bb0;text-align: left;padding: 8px;"> {note} </td>\
    </tr>\
    </table>\
    </div><footer><p>This report is generated on {time_} from {IPAddr}</p><p> ©️AkhilATC</p></footer>"""

    return html_


def convert_html_to_pdf(source_html, output_filename):
    # open output file for writing (truncated binary)
    result_file = open(output_filename, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            source_html,                # the HTML to convert
            dest=result_file)           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    # return False on success and True on errors
    return pisa_status.err


if __name__ == "__main__":
    file_name = f"_bill_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pdf"
    convert_html_to_pdf(__generate_pdf_attributes("A","B","C","D","G","F","F","g","h"),file_name)