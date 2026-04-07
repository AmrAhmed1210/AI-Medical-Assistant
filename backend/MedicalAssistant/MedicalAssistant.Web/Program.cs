using MedicalAssistant.Persistance.Data.DbContexts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Web.Extensions;
using Microsoft.EntityFrameworkCore;

namespace MedicalAssistant;

public class Program
{
    public static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        builder.Services.AddControllers();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddSwaggerGen();

        builder.Services.AddDbContext<MedicalAssistantDbContext>(options =>
            options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

        builder.Services.AddPatientModule();

        builder.Services.AddScoped<IUnitOfWork, UnitOfWork>();
        builder.Services.AddScoped<IAdminRepository, AdminRepository>();
        builder.Services.AddScoped<IAuthService, AuthService>();
        builder.Services.AddScoped<IDoctorService, DoctorService>();
        builder.Services.AddScoped<IReviewService, ReviewService>();
        builder.Services.AddScoped<IAdminService, AdminService>();

        // AutoMapper (single registration)
        builder.Services.AddAutoMapper(
            typeof(DoctorService).Assembly,
            typeof(AdminService).Assembly
        );

        builder.Services.AddCors(options =>
        {
            options.AddPolicy("AllowAll", policy =>
                policy.AllowAnyOrigin()
                      .AllowAnyMethod()
                      .AllowAnyHeader());
        });

        var app = builder.Build();

        if (app.Environment.IsDevelopment())
        {
            app.UseSwagger();
            app.UseSwaggerUI();
        }

        app.UseHttpsRedirection();
        app.UseCors("AllowAll");

        app.UseAuthentication();
        app.UseAuthorization();

        app.MapControllers();

        app.Run();
    }
}