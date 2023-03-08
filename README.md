# venue_popularity

The jupyter notebook contains the code for a Venue Popularity Predictor that uses Yelp's dataset for venue popularities to predict the popularity for new locations.
> The dataset is available under `./data/yelp_dataset.json.gz`.

The model is developed in `scikit-learn` and is a combination of four models that use:
- city: `city_ave`
- category: Type of food served
- location: (`latitude`,`longitude`)
- other attributes: `attire`, `ambience`, etc. 

to predict the venue popularity. The hyperparameters were optimized using `gridsearchCV`.

The full model combines the 4 models mentioned above. 


```python

city_trans = ModelTransformer(city_model)
city_latlong = ModelTransformer(model_latlong)
city_cat = ModelTransformer(model_cat)
city_attr = ModelTransformer(attribute_model)


model_union = FeatureUnion([('city', city_trans),
                      ('location', city_latlong),
                      ('category', city_cat),
                      ('attribute', city_attr)
    ])

full_model=Pipeline([('model_union', model_union), ('lr', LinearRegression())])

```
