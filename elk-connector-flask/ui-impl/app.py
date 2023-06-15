from flask import Flask, redirect, url_for, request,render_template,send_file, flash
import pdf_generator as pdf_gen
import os

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/', methods=['GET'])
def login():
    return render_template('index.html')


@app.route('/gen_report', methods=['POST'])
def gen_report():
    clinic_name = request.form.get('clinic-name') or None
    clinic_logo = request.form.get('clinic-logo') or None

    # physical's info
    physician_name = request.form.get('physician-name') or None
    physician_contact = request.form.get('physician-contact') or None

    # patient's info

    patient_fname = request.form.get('p-name') or None
    patient_lname = request.form.get('p-lname') or None

    patient_dob = request.form.get('p-dob') or None
    patient_contact = request.form.get('p-contact') or None
    # notes
    c_note = request.form.get('input-message') or None
    con_note = request.form.get('input-message-2') or None
    check_list = [clinic_name, physician_name, physician_contact, patient_fname,
                  patient_lname, patient_dob, patient_contact, c_note, con_note]
    print(check_list)
    if any(i is None for i in check_list):
        print('validation error')
        error = "Validation failed : Mandatory fields not allowed to be empty"
        return render_template('index.html', error=error)
    final_name = f"{patient_fname} {patient_lname}"
    resp = pdf_gen.__generate_pdf_attributes(
        clinic_name,
        clinic_logo,
        physician_name,
        physician_contact,
        final_name,
        patient_dob,
        patient_contact,
        c_note,
        con_note

    )
    file_name = f"CR_{patient_fname}_{patient_lname}_{patient_dob}.pdf".replace(" ", "_")
    is_error = pdf_gen.convert_html_to_pdf(resp, file_name)
    if is_error:
        return redirect(url_for("/"))
    else:
        return send_file(file_name, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
