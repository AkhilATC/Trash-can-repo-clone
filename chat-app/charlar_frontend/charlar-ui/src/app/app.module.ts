import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { FormsModule } from '@angular/forms';
import { ChandlerViewComponent } from './chandler-view/chandler-view.component';
import { Routes, RouterModule } from '@angular/router';
import { LoginViewComponent } from './login-view/login-view.component';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { SignupComponent } from './signup/signup.component';
import { JwtInterceptor } from './_helper/jwt.interceptor';
import {MatCardModule} from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import {MatToolbarModule} from '@angular/material/toolbar'; 
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ReactiveFormsModule } from '@angular/forms';




 
@NgModule({
  declarations: [
    AppComponent,
    ChandlerViewComponent,
    LoginViewComponent,
    SignupComponent,
    
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    MatCardModule,
    MatToolbarModule,
    MatInputModule,
    MatFormFieldModule,
    BrowserAnimationsModule,
    ReactiveFormsModule
    
  ],
  providers: [{ provide: HTTP_INTERCEPTORS, useClass: JwtInterceptor, multi: true }],
  bootstrap: [AppComponent]
})
export class AppModule { }
