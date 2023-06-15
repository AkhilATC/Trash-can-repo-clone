import { AuthenticationServiceService } from './../_service/authentication-service.service';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { first } from 'rxjs/operators';

@Component({
  selector: 'app-login-view',
  templateUrl: './login-view.component.html',
  styleUrls: ['./login-view.component.scss']
})
export class LoginViewComponent implements OnInit {
  
  public userName:string;
  public passWord: string;
  public msg:string;
  constructor( private _router: Router,private authenticationServiceService:AuthenticationServiceService) {

    if (this.authenticationServiceService.currentUserValue) {
      this._router.navigate(['/chandler']);
  }
   }
  ngOnInit(): void {
  }

  public logIn(){
    console.log(this.passWord,this.userName);
    this.authenticationServiceService.login(this.userName,this.passWord)
    .pipe(first())
    .subscribe(
      data => {
          this._router.navigate(['/chandler']);
      },
      error => {
          console.log(error)
          this.msg = error.error.message;
      });
   

  }

}
