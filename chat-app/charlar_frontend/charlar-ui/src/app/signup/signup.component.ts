import { ChatServiceService } from './../_service/chat-service.service';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthenticationServiceService } from '../_service/authentication-service.service';
import { first } from 'rxjs/operators';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.scss']
})
export class SignupComponent implements OnInit {

    registerForm: FormGroup;
    loading = false;
    submitted = false;
    
    msg:any;

    constructor(
        private formBuilder: FormBuilder,
        private router: Router,
        private authenticationService: AuthenticationServiceService,
        private _chatService :ChatServiceService
        
    ) {
        // redirect to home if already logged in
        if (this.authenticationService.currentUserValue) {
            this.router.navigate(['/']);
        }
    }

    ngOnInit() {
        this.registerForm = this.formBuilder.group({
            firstName: ['', Validators.required],
            lastName: ['', Validators.required],
            username: ['', Validators.required],
            password: ['', [Validators.required, Validators.minLength(6)]]
        });
    }

    // convenience getter for easy access to form fields
    get f() { return this.registerForm.controls; }

    onSubmit() {
        this.submitted = true;

        // ['name', 'first_name', 'last_name', 'password']

        // stop here if form is invalid
        if (this.registerForm.invalid) {
            return;
        }

        this.loading = true;
        this._chatService.register(this.registerForm.value)
            .pipe(first())
            .subscribe(
                data => {
                    this.msg = data['message']
                    this.router.navigate(['/login']);
                },
                error => {
                    console.log("Failed signUp")
                    this.msg = "SignUp failed ....!!!!"
                    this.loading = false;
                });
    }

}
