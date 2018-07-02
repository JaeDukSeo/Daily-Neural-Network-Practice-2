[![PHP Censor](http://ci.php-censor.info/build-status/image/2?branch=master&label=PHPCensor&style=flat-square)](http://ci.php-censor.info/build-status/view/2?branch=master)
[![Travis CI](https://img.shields.io/travis/php-censor/php-censor/master.svg?label=TravisCI&style=flat-square)](https://travis-ci.org/php-censor/php-censor?branch=master)
[![SensioLabs Insight](https://img.shields.io/sensiolabs/i/26f28bee-a861-45b2-bc18-ed2ac7defd22.svg?label=Insight&style=flat-square)](https://insight.sensiolabs.com/projects/26f28bee-a861-45b2-bc18-ed2ac7defd22)
[![Codecov](https://img.shields.io/codecov/c/github/php-censor/php-censor.svg?label=Codecov&style=flat-square)](https://codecov.io/gh/php-censor/php-censor)
[![Latest Version](https://img.shields.io/packagist/v/php-censor/php-censor.svg?label=Version&style=flat-square)](https://packagist.org/packages/php-censor/php-censor)
[![Total downloads](https://img.shields.io/packagist/dt/php-censor/php-censor.svg?label=Downloads&style=flat-square)](https://packagist.org/packages/php-censor/php-censor)
[![License](https://img.shields.io/packagist/l/php-censor/php-censor.svg?label=License&style=flat-square)](https://packagist.org/packages/php-censor/php-censor)
   
   
<p align="center">
    <img width="800" height="auto" src="/image/image.png" alt="PHP Censor" />
</p>
   
   
**Daily-Neural-Network-Practice-2** This is my repo for daily practice of coding as well as machine learning (Season 2)
The cover image is from this [website](https://www.pexels.com/photo/selective-focus-photography-of-sparkler-955792/). 

**Twitter**: [@JaeDukSeo](https://twitter.com/JaeDukSeo?lang=en).

[![Dashboard](docs/screenshots/dashboard.png)](docs/screenshots/dashboard.png)

More [screenshots](docs/en/screenshots.md).

* [Updating](#updating)
* [Configuring project](#configuring-project)
* [Migrations](#migrations)
* [Tests](#tests)
* [Documentation](#documentation)
* [License](#license)

## System requirements

* Python 3 or 2
* Tensorflow

## Features


* Set up and tear down database tests for [PostgreSQL](docs/en/plugins/pgsql.md), [MySQL](docs/en/plugins/mysql.md) or 
[SQLite](docs/en/plugins/sqlite.md);

* Install [Composer](docs/en/plugins/composer.md) dependencies;

* Run tests for PHPUnit, Atoum, Behat, Codeception and PHPSpec;

* Check code via Lint, PHPParallelLint, Pdepend, PHPCodeSniffer, PHPCpd, PHPCsFixer, PHPDocblockChecker, PHPLoc, 
PHPMessDetect, PHPTalLint and TechnicalDept;

* Run through any combination of the other [supported plugins](docs/en/README.md#plugins), including Campfire, 
CleanBuild, CopyBuild, Deployer, Env, Git, Grunt, Gulp, PackageBuild, Phar, Phing, Shell and Wipe;

* Send notifications on Email, XMPP, Slack, IRC, Flowdock, HipChat and 
[Telegram](https://github.com/LEXASOFT/PHP-Censor-Telegram-Plugin);

* Use your LDAP-server for authentication;

## Installing

* If you don't have python installed please install it. 

* [Add a virtual host to your web server](docs/en/virtual_host.md), pointing to the `public` directory within your new
PHP Censor directory. You'll need to set up rewrite rules to point all non-existent requests to PHP Censor;

* [Set up the PHP Censor Worker](docs/en/workers/worker.md) (Need configured Queue) or 
[a cron-job](docs/en/workers/cron.md) to run PHP Censor builds;

## Installing via Docker

If you want to install PHP Censor as Docker container, you can use 
[php-censor/docker-php-censor](https://github.com/php-censor/docker-php-censor) project.

## Updating
* I am going to make thsi repo my main repo

## Configuring project
* Add project without any project config (Runs "zero-config" plugins, including: Composer, TechnicalDept, PHPLoc, 
PHPCpd, PHPCodeSniffer, PHPMessDetect, PHPDocblockChecker, PHPParallelLint, PHPUnit and Codeception);

* Similar to [Travis CI](https://travis-ci.org), to support PHP Censor in your project, you simply need to add a 
`.php-censor.yml` (`phpci.yml`/`.phpci.yml` for backward compatibility with PHPCI) file to the root of your repository;

* Add project config in PHP Censor project page (And it will cancel file config from project repository);

The project config should look something like this:
More details about [configuring project](docs/en/configuring_project.md).

## Migrations

Run to apply latest migrations:

```bash
cd /path/to/php-censor
./bin/console php-censor-migrations:migrate
```

Run to create new migration:

```bash
cd /path/to/php-censor
./bin/console php-censor-migrations:create NewMigrationName
```

## Tests

```bash
cd /path/to/php-censor

./vendor/bin/phpunit --configuration ./phpunit.xml --coverage-html ./tests/runtime/coverage -vvv --colors=always
```

For Phar plugin tests set 'phar.readonly' setting to Off (0) in `php.ini` config. Otherwise tests will be skipped.  

For database B8Framework tests create empty 'b8_test' database on 'localhost' with user/password: `root/<empty>` 
for MySQL and with user/password: `postgres/<empty>` for PostgreSQL (You can change default test user, password and 
database name in `phpunit.xml` config constants). If connection failed tests will be skipped.

## Documentation

[Full PHP Censor documentation](docs/en/README.md).

## License

PHP Censor is open source software licensed under the [BSD-2-Clause license](LICENSE).