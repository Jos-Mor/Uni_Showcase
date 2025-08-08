---------- remoção de tabelas, se existirem

drop table cupoes cascade constraints;
drop table pessoas cascade constraints;
drop table empregados cascade constraints;
drop table tecnicos cascade constraints;
drop table distribuidores cascade constraints;
drop table clientes cascade constraints;
drop table categorias cascade constraints;
drop table produtos cascade constraints;
drop table veiculos cascade constraints;
drop table encomendas cascade constraints;
drop table tickets cascade constraints;
drop table adquirem cascade constraints;
drop table trabalha_com cascade constraints;
drop table atendem cascade constraints;
drop table contem cascade constraints;
drop table conduzem cascade constraints;
drop table transportado_por cascade constraints;
drop table entregues cascade constraints;

---------- criação das tabelas

create table categorias (
    idCategoria number(10,0),
    nomeCategoria varchar2(50) not null unique,

    primary key (idCategoria)
);

create table cupoes (
    idCupao number(10,0),
    precoPontos number(5,0) not null,
    percentagemDesconto number(4,2) not null,
    data_inicio date not null,
    data_fim date not null,
    idCategoria number(10,0),

    primary key (idCupao),
    foreign key (idCategoria) references categorias(idCategoria),
    
    check (precoPontos > 0),
    check (percentagemDesconto between 0.0 and 100.0)
);

create table pessoas (
    email varchar2(40),
    nome varchar2(50) not null,
    nif number(9,0) not null,
    morada varchar2(50) not null,

    primary key (email)
);

create table empregados (
    email varchar2(40),

    primary key (email),
    foreign key (email) references pessoas(email)
);

create table tecnicos (
    email varchar2(40),

    primary key (email),
    foreign key (email) references empregados(email)
);

create table distribuidores (
    email varchar2(40),

    primary key (email),
    foreign key (email) references empregados(email)
);

create table clientes (
    email varchar2(40),
    passwordCliente varchar2(128) not null,
    carteiraPontos number(5,0) default 0 not null,

    primary key (email),
    foreign key (email) references pessoas(email),
    
    check (carteiraPontos >= 0)
);

create table produtos (
    idProduto number(10,0), 
    nomeProduto varchar2(50) not null unique,
    peso number(6,3) not null, 
    stock number(3,0) not null, 
    preco number(7,2) not null, 
    descricao varchar2(2000) not null, 
    idCategoria number(10,0) not null,

    primary key (idProduto),
    foreign key (idCategoria) references categorias(idCategoria),
    
    check (peso > 0.0),
    check (stock >= 0),
    check (preco > 0.0)
);

create table veiculos (
    matricula varchar(8),
    capacidade number(7,3) not null,

    primary key (matricula)
);

create table encomendas (
    idEncomenda number(10,0),
    dataEncomenda date not null,
    estadoEncomenda varchar2(13) default 'processamento' not null,
    email varchar2(40),

    primary key (idEncomenda),
    foreign key (email) references clientes(email),
    
    check (estadoEncomenda in ('processamento', 'expedido', 'entregue'))
);

create table tickets (
    idTicket number(10,0),
    estadoTicket varchar2(7) default 'aberto' not null,
    descricao varchar2(2000) not null,
    idCategoria number(10,0),
    email varchar2(40),

    primary key (idTicket),
    foreign key (idCategoria) references categorias(idCategoria),
    foreign key (email) references clientes(email),
    
    check (estadoTicket in ('aberto', 'fechado'))
);

create table entregues (
    matricula varchar2(8),
    idEncomenda number(10,0),
    data_entrega date default sysdate,

    primary key (matricula, idEncomenda, data_entrega),
    foreign key (matricula) references veiculos(matricula),
    foreign key (idEncomenda) references encomendas(idEncomenda)
);

create table adquirem (
    email varchar2(40),
    idCupao number(10,0),

    primary key (email, idCupao),
    foreign key (email) references clientes(email),
    foreign key (idCupao) references cupoes(idCupao)
);

create table trabalha_com (
    idCategoria number(10,0), 
    email varchar2(40),

    primary key (idCategoria, email),
    foreign key (idCategoria) references categorias(idCategoria),
    foreign key (email) references tecnicos(email)
);

create table atendem (
    email varchar2(40), 
    idTicket number(10,0),

    primary key (email, idTicket),
    foreign key (email) references tecnicos(email),
    foreign key (idTicket) references tickets(idTicket)
);

create table contem (
    idProduto number(10,0), 
    idEncomenda number(10,0), 
    quantidade number(3,0) not null,

    primary key (idProduto, idEncomenda),
    foreign key (idProduto) references produtos(idProduto),
    foreign key (idEncomenda) references encomendas(idEncomenda),
    
    check (quantidade > 0)
);

create table conduzem (
    email varchar2(40),
    matricula varchar(8),

    primary key (email),
    foreign key (email) references distribuidores(email),
    foreign key (matricula) references veiculos(matricula)
);

create table transportado_por (
    idEncomenda number(10,0),
    matricula varchar(8),

    primary key (idEncomenda),
    foreign key (idEncomenda) references encomendas(idEncomenda),
    foreign key (matricula) references veiculos(matricula)
);

---------- garantias de integridade com triggers

-- garante que o cliente tem pontos suficientes para comprar um cupao
create or replace trigger comprar_cupao
    before insert on adquirem
    for each row
        declare 
            cl_pontos number;
            cp_preco number;
            resultado number;
        begin
            select carteiraPontos into cl_pontos 
            from clientes 
            where clientes.email = :new.email;
    
            select precoPontos into cp_preco
            from cupoes
            where cupoes.idCupao = :new.idCupao;
    
            resultado := cl_pontos - cp_preco;
    
            if (resultado >= 0) then
                update clientes
                set carteiraPontos = resultado
                where clientes.email = :new.email;
            else 
                Raise_Application_Error (
                    -20100, 'Cliente não tem pontos suficientes para comprar o cupão - ' ||
                            'Carteira: ' || to_char(cl_pontos) || 
                            '; Preço: ' || to_char(cp_preco)
                );
            end if;
        end;
/

-- verifica que cada empregado não pode ser 
-- distribuidor e técnico ao mesmo tempo.
create or replace trigger inserir_tecnico
    before insert on tecnicos
    for each row
        declare r number(1);
        begin
            select count(*) into r
            from distribuidores
            where email = :new.email;
    
            if (r > 0) then
                Raise_Application_Error (-20100, 'Empregado já está inserido como distribuidor');
            end if;
        end;
/
create or replace trigger inserir_distribuidor
    before insert on distribuidores
    for each row
        declare r number(1);
        begin
            select count(*) into r
            from tecnicos
            where email = :new.email;
    
            if (r > 0) then
                Raise_Application_Error (-20100, 'Empregado já está inserido como técnico');
            end if;
        end;
/

-- verifica que cada técnico irá responder apenas aos tickets 
-- que pertencem a uma das categorias nas quais estão encarregues.
create or replace trigger responder_ticket
    before insert on atendem
    for each row
        declare 
                categoriaTicket number;
                ocorreCategoria number;
        begin
            select idCategoria into categoriaTicket
            from tickets 
            where :new.idTicket = tickets.idTicket;
                        
            select count(*) into ocorreCategoria
            from trabalha_com
            where categoriaTicket = trabalha_com.idCategoria 
                and trabalha_com.email = :new.email;
                
            if (ocorreCategoria = 0) then
                Raise_Application_Error (
                    -20100, 'Técnico não pertence à categoria do ticket (' || categoriaTicket || ')'
                );
            end if;    
        end;
/

-- verifica se a quantidade de produtos de uma dada encomenda 
-- não excede o stock. Caso respeite esta condição, então será 
-- retirada essa quantidade ao stock do produto.
create or replace trigger stock_produto
    before insert on contem
    for each row
        declare 
            stockProduto number;
            resultado number;
        begin
            select stock into stockProduto
            from produtos 
            where :new.idProduto = produtos.idProduto;
            
            resultado := stockProduto - :new.quantidade;
            
            if (resultado >= 0) then
                update produtos
                set stock = resultado
                where produtos.idProduto = :new.idProduto;
            else
                Raise_Application_Error (
                -20100, 'Quantidade excede o stock do produto - ' || 
                        'Stock: ' || to_char(stockProduto) || 
                        '; Quantidade: ' || :new.quantidade 
            );
            end if;
        end;
/

-- verifica se há capacidade disponível no veículo 
-- para poder transportar uma dada encomenda.
create or replace trigger peso_disponivel
    before insert on contem
    for each row
        declare
            veiculo_atribuido number;
            veiculo varchar2(8);
            p_total number;
            c_veiculo number;
            p_disponivel number;
            p_encomenda number;
            p_total_encomenda number;
        begin
            select count(*) into veiculo_atribuido
            from transportado_por
            where :new.idEncomenda = transportado_por.idEncomenda;
                        
            if (veiculo_atribuido > 0) then
                select matricula into veiculo
                from transportado_por
                where :new.idEncomenda = transportado_por.idEncomenda;
                
                with encomendas_pesos as (
                    select idEncomenda, quantidade * peso p_por_encomenda
                    from encomendas inner join transportado_por using(idEncomenda)
                                    inner join contem using(idEncomenda)
                                    inner join produtos using(idProduto)
                    where idEncomenda <> :new.idEncomenda and
                        matricula = veiculo
                )
                select sum(p_por_encomenda) into p_total
                from encomendas_pesos;
                
                select capacidade into c_veiculo
                from veiculos
                where matricula = veiculo;
                
                -- nvl é necessário pois sum(p_por_encomenda) pode retornar null
                p_disponivel := c_veiculo - nvl(p_total, 0);
                
                select peso into p_encomenda 
                from produtos
                where idProduto = :new.idProduto;
                        
                p_total_encomenda := p_encomenda * :new.quantidade;         
                  
                if (p_total_encomenda > p_disponivel) then
                    Raise_Application_Error (
                        -20100, 'Veículo '|| veiculo || ' está cheio - ' ||
                                'Peso disponível: ' || to_char(p_disponivel) || 
                                '; Pretendido: ' || to_char(p_total_encomenda) 
                    );
                end if;
            end if;
        end;
/
create or replace trigger peso_disponivel_com_quantidade
    before insert on transportado_por
    for each row
        declare
            ja_com_quantidade number;
            veiculo varchar2(8);
            p_total number;
            c_veiculo number;
            p_disponivel number;
            p_total_encomenda number;
        begin
            select count(*) into ja_com_quantidade
            from transportado_por
            where :new.idEncomenda = transportado_por.idEncomenda;

            if (ja_com_quantidade > 0) then
                select matricula into veiculo
                from transportado_por
                where :new.idEncomenda = transportado_por.idEncomenda;

                with encomendas_pesos as (
                    select idEncomenda, quantidade * peso p_por_encomenda
                    from encomendas inner join transportado_por using(idEncomenda)
                                    inner join contem using(idEncomenda)
                                    inner join produtos using(idProduto)
                    where idEncomenda <> :new.idEncomenda and
                        matricula = veiculo
                )
                select sum(p_por_encomenda) into p_total
                from encomendas_pesos;
                
                select capacidade into c_veiculo
                from veiculos
                where matricula = veiculo;
                
                -- nvl é necessário pois sum(p_por_encomenda) pode retornar null
                p_disponivel := c_veiculo - nvl(p_total, 0);
                
                select quantidade * peso into p_total_encomenda 
                from contem inner join produtos using(idProduto)
                where idEncomenda = :new.idEncomenda;
                
                if (p_total_encomenda > p_disponivel) then
                    Raise_Application_Error (
                        -20100, 'Veículo '|| veiculo || ' está cheio - ' ||
                                'Peso disponível: ' || to_char(p_disponivel) || 
                                '; Pretendido: ' || to_char(p_total_encomenda) 
                    );
                end if;
            end if;
        end;
/


-- garante que antes da alteração do estado da encomenda é
-- necessário que a mesma tenha um veículo atribuído.
create or replace trigger estado_encomenda
    before update of estadoEncomenda on encomendas
    for each row
        declare 
            r number(1);
        begin
          if (:new.estadoEncomenda = 'expedido' or :new.estadoEncomenda = 'entregue') then
            select count(*) into r
            from transportado_por 
            where idEncomenda = :new.idEncomenda;
    
            if (r = 0) then
                Raise_Application_Error (
                    -20100, 'Não é possível atualizar estado de encomenda para ' || 
                            :new.estadoEncomenda || ', visto que não tem um veículo atribuído'
                );
            end if;
          end if;
        end;
/

-- retira da tabela transportado_por uma dada encomenda,
-- caso o seu estado tenha sido mudado para 'entregue'.
create or replace trigger encomenda_entregue
    after update of estadoEncomenda on encomendas
    for each row
        begin
            if (:new.estadoEncomenda = 'entregue') then
                insert into entregues (matricula, idEncomenda)
                    select matricula, idEncomenda
                    from transportado_por
                    where transportado_por.idEncomenda = :new.idEncomenda;

                delete from transportado_por
                where transportado_por.idEncomenda = :new.idEncomenda;
            end if;
        end;
/

-- garante que não se insere uma encomenda na tabela transportado_por 
-- sendo que já se encontra na tabela entregues.
create or replace trigger ja_entregue
    before insert on transportado_por
    for each row
        declare 
            r number(1);
        begin
            select count(*) into r
            from entregues
            where :new.idEncomenda = idEncomenda;
            
            if (r > 0) then
                Raise_Application_Error (
                    -20100, 'Encomenda ' || to_char(:new.idEncomenda) || ' já se encontra entregue');
            end if;
        end;
/        

---------- geradores de ids

drop sequence seq_id_categoria;
create sequence seq_id_categoria
start with 1
increment by 1;

drop sequence seq_id_produto;
create sequence seq_id_produto
start with 1
increment by 1;

drop sequence seq_id_ticket;
create sequence seq_id_ticket
start with 1
increment by 1;

drop sequence seq_id_encomenda;
create sequence seq_id_encomenda
start with 1
increment by 1;

drop sequence seq_id_cupao;
create sequence seq_id_cupao
start with 1
increment by 1;

---------- inserção de dados nas tabelas

insert into pessoas values ('joao@gmail.com', 'João', 233423432, 'rua do ganhão nº 13');
insert into pessoas values ('ze@gmail.com', 'Zé', 423423423, 'rua D. Manuel Lisboa nº 21');
insert into pessoas values ('francisco@gmail.com', 'Francisco', 233423432, 'rua S. João, Almada nº 21');
insert into pessoas values ('rodrigo@gmail.com', 'Rodrigo', 456754312, 'rua Sá Pessoa, Coimbra nº 34');
insert into pessoas values ('david@gmail.com', 'David', 765757567, 'rua Batista, Vila Nova de Gaia nº 24');
insert into pessoas values ('joel@gmail.com', 'Joel', 424265756, 'rua Luis Manuel, Elvas nº 12');
insert into pessoas values ('cabrita@gmail.com', 'Cabrita', 120894564, 'rua do Conde, Leiria nº 32');
insert into pessoas values ('socrates@gmail.com', 'Socrates', 436509807, 'rua das Aguias, Torres Vedras nº 53');
insert into pessoas values ('ventura@gmail.com', 'Ventura', 421213123, 'rua Judas, Fátima nº 1');
insert into pessoas values ('portas@gmail.com', 'Portas', 957434534, 'rua Paco Bandeira, Elvas nº 98');
insert into pessoas values ('rendeiro@gmail.com', 'Rendeiro', 453245345, 'rua Manuel II, Porto nº 64');
insert into pessoas values ('sousa@gmail.com', 'Sousa', 703487526, 'rua D. Rui I, Borba nº 31');

insert into empregados values('joao@gmail.com');
insert into empregados values('ze@gmail.com');
insert into empregados values('francisco@gmail.com');
insert into empregados values('rodrigo@gmail.com');
insert into empregados values('cabrita@gmail.com');
insert into empregados values('socrates@gmail.com');
insert into empregados values('ventura@gmail.com');
insert into empregados values('portas@gmail.com');

insert into tecnicos values('joao@gmail.com');
insert into tecnicos values('ze@gmail.com');
insert into tecnicos values('cabrita@gmail.com');
insert into tecnicos values('socrates@gmail.com');

insert into distribuidores values('francisco@gmail.com');
insert into distribuidores values('rodrigo@gmail.com');
insert into distribuidores values('ventura@gmail.com');
insert into distribuidores values('portas@gmail.com');

insert into clientes(email, passwordCliente) values('david@gmail.com', '1234');
insert into clientes(email, passwordCliente) values('joel@gmail.com', '6577');
insert into clientes(email, passwordCliente) values('rendeiro@gmail.com', '4534');
insert into clientes(email, passwordCliente) values('sousa@gmail.com', '3423');

insert into categorias values (seq_id_categoria.nextval, 'Imóveis');
insert into categorias values (seq_id_categoria.nextval, 'Eletrónica');
insert into categorias values (seq_id_categoria.nextval, 'Gaming');
insert into categorias values (seq_id_categoria.nextval, 'Música');
insert into categorias values (seq_id_categoria.nextval, 'Vestuário');
insert into categorias values (seq_id_categoria.nextval, 'Desporto');
insert into categorias values (seq_id_categoria.nextval, 'Bricolage');
insert into categorias values (seq_id_categoria.nextval, 'Jardim');
insert into categorias values (seq_id_categoria.nextval, 'Limpeza');

insert into trabalha_com values(1, 'cabrita@gmail.com');
insert into trabalha_com values(2, 'cabrita@gmail.com');
insert into trabalha_com values(3, 'cabrita@gmail.com');
insert into trabalha_com values(2, 'ze@gmail.com');
insert into trabalha_com values(1, 'socrates@gmail.com');
insert into trabalha_com values(4, 'ze@gmail.com');
insert into trabalha_com values(5, 'joao@gmail.com');
insert into trabalha_com values(7, 'joao@gmail.com');
insert into trabalha_com values(3, 'socrates@gmail.com');

insert into cupoes values (seq_id_cupao.nextval, 20, 20.0,  to_date('2022.03.05','YYYY.MM.DD'), to_date('2022.08.10','YYYY.MM.DD'), 1);
insert into cupoes values (seq_id_cupao.nextval, 30, 25.0,  to_date('2022.02.10','YYYY.MM.DD'), to_date('2022.09.15','YYYY.MM.DD'), 2);
insert into cupoes values (seq_id_cupao.nextval, 45, 50.0,  to_date('2022.04.17','YYYY.MM.DD'), to_date('2022.10.19','YYYY.MM.DD'), 3);
insert into cupoes values (seq_id_cupao.nextval, 35, 30.0,  to_date('2022.05.20','YYYY.MM.DD'), to_date('2022.11.20','YYYY.MM.DD'), 4);
insert into cupoes values (seq_id_cupao.nextval, 50, 55.0,  to_date('2022.01.30','YYYY.MM.DD'), to_date('2022.07.23','YYYY.MM.DD'), 5);

insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'Blue Screen of Death em Laptop Asus', 2, 'david@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'Skate vinha sem rodas', 6, 'joel@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'A sola ténis que comprei descoseu-se', 6, 'rendeiro@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'A minha consola não detecta o jogo', 3, 'joel@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'Um dos pés da minha mesa de cabeceira lascou-se', 1, 'joel@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'O micro-ondas deixou de funcionar', 2, 'sousa@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'A blusa reasgou-se', 5, 'sousa@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'A lata de tinta tem o rótulo errado', 7, 'joel@gmail.com');
insert into tickets (idTicket, descricao, idCategoria, email) values (seq_id_ticket.nextval, 'Um dos pneus da minha bicicleta está furado', 6, 'rendeiro@gmail.com');

insert into produtos values(seq_id_produto.nextval, 'Headphones', 0.3, 900, 50.32, 'Headphones 7.1', 3);
insert into produtos values(seq_id_produto.nextval, 'Mesa', 5.0, 350, 100.21, 'Mesa de vidro', 1);
insert into produtos values(seq_id_produto.nextval, 'Computador', 3.0, 100, 700.23, 'Computador gaming', 3);
insert into produtos values(seq_id_produto.nextval, 'Tenis', 0.6, 800, 60.43, 'Ténis para correr', 6);
insert into produtos values(seq_id_produto.nextval, 'Camisa', 0.2, 650, 40.54, 'Camisa branca', 5);
insert into produtos values(seq_id_produto.nextval, 'Guitarra', 3.2, 300, 250.21, 'Guitarra elétrica', 4);
insert into produtos values(seq_id_produto.nextval, 'Cama', 15.0, 123, 300.23, 'Camisa branca', 1);
insert into produtos values(seq_id_produto.nextval, 'Tinta Branca', 8.0, 123, 34.43, 'Tinta para parede', 7);
insert into produtos values(seq_id_produto.nextval, 'Tábua de madeira', 6.0, 34, 25.12, 'Tábua em bruto 120x150x3 (cm)', 7);
insert into produtos values(seq_id_produto.nextval, 'Berbequim', 1.5, 67, 50.6, 'Berbequim sem fios', 7);
insert into produtos values(seq_id_produto.nextval, 'Micro-ondas', 4.0, 78, 79.21, 'Micro-ondas 25L', 2);
insert into produtos values(seq_id_produto.nextval, 'Corta Relvas', 6.0, 23, 134.43, 'Corta Relvas 1200w 6kg', 8);
insert into produtos values(seq_id_produto.nextval, 'Sopadror', 3.2, 5, 42.21, 'Soprador de Folhas 700W 13000Rpm', 8);
insert into produtos values(seq_id_produto.nextval, 'Limpa Vidros', 0.4, 335, 6.40, 'Limpa Vidros Magnético', 9);
insert into produtos values(seq_id_produto.nextval, 'Esfregona e Balde', 2.5, 68, 29.90, 'Esfregona e Balde Vileda', 9);


-- necessário para não disparar 
-- o trigger compra_ticket.
update clientes
set carteiraPontos = 300
where email in (
    'joel@gmail.com', 'rendeiro@gmail.com', 
    'sousa@gmail.com', 'david@gmail.com'
);

insert into adquirem values('joel@gmail.com', 1);
insert into adquirem values('david@gmail.com', 2);
insert into adquirem values('rendeiro@gmail.com', 3);
insert into adquirem values('sousa@gmail.com', 4);
insert into adquirem values('david@gmail.com', 5);

insert into atendem values('cabrita@gmail.com', 4);
insert into atendem values('cabrita@gmail.com', 5);
insert into atendem values('cabrita@gmail.com', 6);
insert into atendem values('ze@gmail.com', 1);
insert into atendem values('socrates@gmail.com', 5);
insert into atendem values('socrates@gmail.com', 4);
insert into atendem values('joao@gmail.com', 7);

update tickets
set estadoTicket = 'fechado'
where idTicket in (
    2, 3, 6, 7, 9
);

insert into veiculos values ('AZ-43-2D', 943.23);
insert into veiculos values ('EW-D2-3G', 932.22);
insert into veiculos values ('DD-1W-V4', 2043.1);
insert into veiculos values ('23-SW-5B', 600.43);

insert into conduzem values ('francisco@gmail.com', 'AZ-43-2D');
insert into conduzem values ('portas@gmail.com', 'EW-D2-3G');
insert into conduzem values ('rodrigo@gmail.com', 'DD-1W-V4');
insert into conduzem values ('ventura@gmail.com', '23-SW-5B');

insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.03.05','YYYY.MM.DD'), 'david@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.02.20','YYYY.MM.DD'), 'rendeiro@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.02.17','YYYY.MM.DD'), 'joel@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.03.13','YYYY.MM.DD'), 'sousa@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.01.03','YYYY.MM.DD'), 'joel@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.02.26','YYYY.MM.DD'), 'david@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.05.3','YYYY.MM.DD'), 'joel@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.04.24','YYYY.MM.DD'), 'rendeiro@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.03.23','YYYY.MM.DD'), 'david@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.05.12','YYYY.MM.DD'), 'joel@gmail.com');
insert into encomendas(idEncomenda, dataEncomenda, email) values (seq_id_encomenda.nextval, to_date('2022.05.12','YYYY.MM.DD'), 'rendeiro@gmail.com');

insert into transportado_por values(1, 'AZ-43-2D');
insert into transportado_por values(2, 'AZ-43-2D');
insert into transportado_por values(3, 'DD-1W-V4');
insert into transportado_por values(4, 'EW-D2-3G');
insert into transportado_por values(5, '23-SW-5B');
insert into transportado_por values(6, 'EW-D2-3G');
insert into transportado_por values(8, '23-SW-5B');
insert into transportado_por values(9, 'DD-1W-V4');
insert into transportado_por values(10, 'DD-1W-V4');
insert into transportado_por values(11, 'AZ-43-2D');

insert into contem values(2, 1, 10);
insert into contem values(3, 2, 1);
insert into contem values(2, 4, 3);
insert into contem values(4, 5, 2);
insert into contem values(5, 6, 5);
insert into contem values(13, 7, 2);
insert into contem values(15, 8, 1);
insert into contem values(15, 9, 3);
insert into contem values(12, 10, 4);
insert into contem values(14, 11, 2);

commit;
