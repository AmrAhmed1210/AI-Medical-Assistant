using MedicalAssistant.Application.Services;
using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Services.MappingProfiles;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Web.Extensions;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // Add services to the container
        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        // Add DbContext
        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
        {
            options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection"));
        });

        // Register Modules
        builder.Services.AddPatientModule();

        // AutoMapper
        // Use this syntax for AutoMapper 13+ in .NET 8
        builder.Services.AddAutoMapper(cfg =>
        {
            cfg.AddProfile<DoctorProfile>();
        }, typeof(DoctorProfile).Assembly);

        // Register Doctor Service
        builder.Services.AddScoped<IDoctorService, DoctorService>();

        // Optional: CORS (allow frontend requests)
        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
            {
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader();
            });
        });

        var app = builder.Build();

        // Configure the HTTP request pipeline
        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseHttpsRedirection();

        app.UseCors("AllowAll"); // <-- enable CORS

        app.UseAuthorization();

        app.MapControllers();

        app.Run();
    }
}