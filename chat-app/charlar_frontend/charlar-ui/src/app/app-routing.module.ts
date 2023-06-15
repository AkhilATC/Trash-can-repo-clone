import { SignupComponent } from './signup/signup.component';
import { AppComponent } from './app.component';
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ChandlerViewComponent } from './chandler-view/chandler-view.component';
import { LoginViewComponent } from './login-view/login-view.component';
import { AuthGuardService } from './_helper/auth-guard.service';

const routes: Routes = [

  { path: 'chandler',component: ChandlerViewComponent},
  { path:'login', component: LoginViewComponent},
  {path:'',component: AppComponent, canActivate: [AuthGuardService]},
  { path: 'register', component: SignupComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
