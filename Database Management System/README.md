# Database Management System Project - GLVO E-commerce Platform

Oracle database system with SQL, triggers, and Oracle APEX web application for an academic e-commerce platform.

This project demonstrates database design and implementation skills through a complete e-commerce database with business rule enforcement and a web interface built using Oracle APEX.

## 🎯 Project Overview

The **GLVO** (Gestão de Loja Virtual Online) system is an e-commerce database platform managing:

- **Customer Management**: User accounts, authentication, and loyalty points
- **Product Catalog**: Categories, inventory, and pricing
- **Order Processing**: Shopping cart, order fulfillment, and delivery tracking
- **Support System**: Ticket management and technical support
- **Logistics**: Vehicle fleet management and delivery coordination
- **Promotions**: Coupon system with category-based discounts

## 📁 Project Structure

### 🗄️ Core Database (GLVO/)
- **`BD35-GLVO-SQL.sql`** - Complete database schema with triggers and constraints
- **`BD35-GLVO-APEX.sql`** - Oracle APEX application export (13 pages)
- **`BD35-GLVO-Relatorio.pdf`** - Project documentation and analysis

### 📋 Additional Components
- **`ProjetoBD.pdf`** - Project specifications and requirements

## 🏗️ Database Architecture

### Core Entities
```sql
-- User Management Hierarchy
PESSOAS → EMPREGADOS → {TECNICOS, DISTRIBUIDORES}
       → CLIENTES

-- Product & Order Management
CATEGORIAS → PRODUTOS → CONTEM ← ENCOMENDAS

-- Support & Logistics
TICKETS ← ATENDEM → TECNICOS
VEICULOS ← CONDUZEM → DISTRIBUIDORES
```

### Database Features
- **18 interconnected tables** with referential integrity
- **10 triggers** for business rule enforcement
- **Comprehensive constraints** ensuring data integrity
- **Sequence generators** for auto-incrementing IDs
- **Normalized schema** reducing data redundancy

## 🚀 Key Technical Implementations

### 1. **Inventory Management**
```sql
-- Automatic stock deduction with validation
CREATE OR REPLACE TRIGGER stock_produto
    BEFORE INSERT ON contem
    FOR EACH ROW
    -- Validates stock availability and updates inventory
```

### 2. **Vehicle Capacity Management**
```sql
-- Weight calculation and vehicle assignment validation
CREATE OR REPLACE TRIGGER peso_disponivel
    -- Ensures orders don't exceed vehicle capacity
    -- Calculates total weight across multiple orders
```

### 3. **Loyalty Points System**
```sql
-- Automated points deduction for coupon purchases
CREATE OR REPLACE TRIGGER comprar_cupao
    -- Validates sufficient points
    -- Automatic balance updates
```

### 4. **Role-Based Access Control**
```sql
-- Ensures employees can't have conflicting roles
CREATE OR REPLACE TRIGGER inserir_tecnico
CREATE OR REPLACE TRIGGER inserir_distribuidor
    -- Prevents dual technician/distributor assignments
```

### 5. **Order Lifecycle Management**
```sql
-- Automated order state transitions
CREATE OR REPLACE TRIGGER estado_encomenda
CREATE OR REPLACE TRIGGER encomenda_entregue
    -- Enforces business rules for order processing
    -- Automatic delivery tracking
```

## 🌐 Oracle APEX Web Application

### Application Features
- **13 responsive pages** with form-based interface
- **Interactive reports** with filtering and sorting
- **Data entry forms** with validation
- **Role-based navigation** and security
- **Dashboard functionality** for data overview

### Key Application Modules
1. **Customer Management** - Registration, profiles, loyalty points
2. **Product Catalog** - Category management, inventory tracking
3. **Order Processing** - Cart functionality, order tracking
4. **Support Tickets** - Issue tracking and resolution
5. **Logistics Dashboard** - Vehicle management, delivery coordination
6. **Reports** - Basic data reporting and analytics

## 📊 Business Logic Implementation

### Database Queries & Operations
- **Customer segmentation** by purchase behavior
- **Inventory tracking** with stock management
- **Vehicle capacity validation** based on weight limits
- **Support ticket management** with technician assignment
- **Order processing** with state management

### Data Integrity Features
- **Referential integrity** across all relationships
- **Business rule validation** through trigger system
- **Constraint-based validation** for data quality
- **Transaction management** ensuring consistency

## 🔧 Technical Implementation

### Database Design
- **Proper normalization** following relational principles
- **Foreign key relationships** maintaining data consistency
- **Check constraints** for data validation
- **Trigger-based business logic** for complex rules

### Application Architecture
- **Oracle APEX framework** for rapid web development
- **SQL-based data access** with proper joins
- **Form-based user interface** for data management
- **Report generation** for business insights

## 🎓 Academic Project Highlights

**Project Demonstrates:**
- **Database design principles** with proper ER modeling
- **SQL implementation** with complex relationships
- **Business rule enforcement** through triggers
- **Web application development** using Oracle APEX
- **Complete project lifecycle** from design to implementation

## 💻 Technologies Used

- **Oracle Database** - Relational database management system
- **SQL** - Data definition, manipulation, and querying
- **PL/SQL** - Triggers and stored logic
- **Oracle APEX** - Web application development platform
- **Database Design** - ER modeling and normalization

---

This project represents a complete academic database solution demonstrating the ability to design, implement, and deploy a functional database system with a web-based user interface for e-commerce operations.