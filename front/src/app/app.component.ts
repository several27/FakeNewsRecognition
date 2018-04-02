import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  predictionsMulticlass = [];
  predictionsBinary = [];

  view: any[] = [700, 400];

  // options
  showXAxis = true;
  showYAxis = true;
  gradient = false;
  explodeSlices = true;
  showLegend = true;
  showXAxisLabel = true;
  xAxisLabel = 'Country';
  showYAxisLabel = true;
  yAxisLabel = 'Population';

  colorScheme = {
    domain: ['#A10A28', '#5AA454', '#C7B42C', '#AAAAAA']
  };

  // line, area
  autoScale = true;

  url = '';
  title = '';
  content = '';

  constructor(private http: HttpClient) {}

  recognise() {
    console.log(this.title);

    this.http.post('http://ec2-35-176-215-209.eu-west-2.compute.amazonaws.com:7070/v1/predict', {
      url: this.url,
      title: this.title,
      content: this.content
    }).subscribe((response: any) => {
      const classes = [];
      for (const c in response.data.classes) {
        if (response.data.classes.hasOwnProperty(c)) {
          classes.push({name: c, value: response.data.classes[c]});
        }
      }

      this.predictionsMulticlass = classes;
      this.predictionsBinary = [
        {name: 'Fake', value: response.data.fake}, 
        {name: 'Real', value: 1 - response.data.fake}
      ];
    });
  }
}
